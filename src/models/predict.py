"""
Prediction tool: wraps the classical repeat-purchase model artifact so the
agent can call it as a third tool alongside SQL and RAG.

Feature source: the model expects nine "first order" features
(first_order_revenue, first_order_quantity, first_order_avg_price,
first_order_unique_products, first_order_lines, and log1p_ versions of the
first four). These are computed LIVE from orders + order_items for a given
customer_id, replicating src/features/engineering.py's
build_repeat_purchase_dataset() exactly:

  * "first order" = the customer's earliest invoice by invoice_date
    (ties broken by invoice number), matching the pandas sort in
    engineering.py — Cancelled orders are NOT excluded here, because
    engineering.py's _clean_columns() does not filter them out either.
    This differs from the is_cancelled = 0 convention used elsewhere in
    the SQL tool / analytics_guidelines.md. That inconsistency is real
    and should be called out in the report, not silently "fixed" here,
    since silently filtering cancellations would make live predictions
    diverge from what the model was actually trained on.
  * first_order_avg_price = AVG(unit_price) across that invoice's line
    items (unweighted mean, matching pandas .agg(("Price", "mean"))).
  * log1p transform is SIGNED: sign(x) * log1p(abs(x)), applied to
    revenue/quantity/avg_price/lines only (not unique_products) —
    matching engineering.py's loop exactly.

Drop this file at: src/models/predict.py
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# The four base columns that get a signed-log1p companion feature, in the
# exact order engineering.py iterates over them.
_LOG1P_BASE_COLUMNS = (
    "first_order_revenue",
    "first_order_quantity",
    "first_order_avg_price",
    "first_order_lines",
)


def _signed_log1p(x: float) -> float:
    sign = -1.0 if x < 0 else (1.0 if x > 0 else 0.0)
    return sign * math.log1p(abs(x))


@dataclass
class PredictionResult:
    prediction: Any
    probability: float | None
    label: str
    features_used: dict[str, Any]
    model_version: str
    status: str = "success"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "probability": self.probability,
            "label": self.label,
            "features_used": self.features_used,
            "model_version": self.model_version,
            "status": self.status,
            "reason": self.reason,
        }


class PredictionTool:
    """Loads a scikit-learn-style model + its metadata json and serves predictions."""

    def __init__(self, model: Any, metadata: dict, db_path: str | Path | None = None):
        self.model = model
        self.metadata = metadata
        self.features: list[str] = (
            metadata.get("features")
            or metadata.get("feature_names")
            or list(getattr(model, "feature_names_in_", []) or [])
        )
        self.target: str = metadata.get("target", "repeat_purchase")
        self.model_version: str = metadata.get("model_version") or metadata.get("version", "v1")
        self.metrics: dict = metadata.get("metrics", {})
        self.positive_label: str = metadata.get("positive_label", "repeat")
        self.negative_label: str = metadata.get("negative_label", "no_repeat")
        self.db_path = db_path

        if not self.features:
            raise ValueError(
                "Could not determine feature list from the metadata json or "
                "the model itself. Add a 'features': [...] key to the "
                "matching .json artifact file."
            )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_artifact(
        cls,
        artifact_base: str | Path,
        db_path: str | Path | None = "data/retail.db",
    ) -> "PredictionTool":
        base = Path(artifact_base)
        model_path = base.with_suffix(".joblib")
        meta_path = base.with_suffix(".json")
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        model = joblib.load(model_path)
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(model=model, metadata=metadata, db_path=db_path)

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        metric_str = ", ".join(f"{k}={v}" for k, v in self.metrics.items()) or "n/a"
        return (
            f"Predicts '{self.target}' for a customer using a trained "
            f"classical model (version {self.model_version}), based only "
            f"on that customer's FIRST observed order. "
            f"Reported offline metrics: {metric_str}. "
            "Use this tool when the question asks to predict, forecast, or "
            "estimate the likelihood of a customer's FUTURE behavior "
            "(e.g. repeat purchase) rather than report a historical "
            "fact (use sql) or a definition (use rag)."
        )

    # ------------------------------------------------------------------ #
    def _fetch_features_for_customer(self, customer_id: str) -> dict[str, Any]:
        """Computes the nine "first order" features live from orders +
        order_items, replicating engineering.py's
        build_repeat_purchase_dataset() for a single customer."""
        if not self.db_path or not Path(self.db_path).exists():
            raise RuntimeError("No database configured to look up customer features.")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # "first order" = earliest invoice_date for this customer,
            # ties broken by invoice_no. Intentionally NOT filtering
            # is_cancelled, to match engineering.py's training behavior.
            first_invoice_row = conn.execute(
                """
                SELECT invoice_no
                FROM orders
                WHERE customer_id = ?
                ORDER BY invoice_date ASC, invoice_no ASC
                LIMIT 1
                """,
                (customer_id,),
            ).fetchone()

            if first_invoice_row is None:
                raise KeyError(f"No orders found for customer_id={customer_id!r}")

            first_invoice_no = first_invoice_row["invoice_no"]

            agg_row = conn.execute(
                """
                SELECT
                    SUM(oi.revenue)                    AS first_order_revenue,
                    SUM(oi.quantity)                   AS first_order_quantity,
                    AVG(oi.unit_price)                 AS first_order_avg_price,
                    COUNT(DISTINCT oi.stock_code)       AS first_order_unique_products,
                    COUNT(*)                            AS first_order_lines
                FROM order_items oi
                WHERE oi.invoice_no = ?
                """,
                (first_invoice_no,),
            ).fetchone()

        if agg_row is None or agg_row["first_order_lines"] == 0:
            raise KeyError(
                f"No order_items found for customer_id={customer_id!r}'s "
                f"first invoice ({first_invoice_no!r})."
            )

        base = {
            "first_order_revenue": float(agg_row["first_order_revenue"] or 0.0),
            "first_order_quantity": float(agg_row["first_order_quantity"] or 0.0),
            "first_order_avg_price": float(agg_row["first_order_avg_price"] or 0.0),
            "first_order_unique_products": float(agg_row["first_order_unique_products"] or 0.0),
            "first_order_lines": float(agg_row["first_order_lines"] or 0.0),
        }
        for col in _LOG1P_BASE_COLUMNS:
            base[f"log1p_{col}"] = _signed_log1p(base[col])

        # Only return what the model actually asks for, in case the
        # metadata's feature list is a subset/superset for some reason.
        return {f: base[f] for f in self.features if f in base}

    # ------------------------------------------------------------------ #
    def predict(
        self,
        features: dict[str, Any] | None = None,
        customer_id: str | None = None,
    ) -> PredictionResult:
        try:
            if features is None:
                if customer_id is None:
                    return PredictionResult(
                        prediction=None, probability=None, label="",
                        features_used={}, model_version=self.model_version,
                        status="error",
                        reason="Need either a feature dict or a customer_id.",
                    )
                features = self._fetch_features_for_customer(customer_id)

            missing = [f for f in self.features if f not in features]
            if missing:
                return PredictionResult(
                    prediction=None, probability=None, label="",
                    features_used=features, model_version=self.model_version,
                    status="error",
                    reason=f"Missing required features: {missing}",
                )

            row = pd.DataFrame([{f: features[f] for f in self.features}])
            pred = self.model.predict(row)[0]
            proba = None
            if hasattr(self.model, "predict_proba"):
                proba = float(self.model.predict_proba(row)[0][-1])

            label = self.positive_label if pred in (1, True, "1") else self.negative_label
            return PredictionResult(
                prediction=pred if not hasattr(pred, "item") else pred.item(),
                probability=proba,
                label=label,
                features_used=features,
                model_version=self.model_version,
            )
        except Exception as exc:  # keep the agent alive on model/data errors
            return PredictionResult(
                prediction=None, probability=None, label="",
                features_used=features or {}, model_version=self.model_version,
                status="error", reason=str(exc),
            )


def get_default_prediction_tool(
    models_dir: str | Path = "models",
    artifact_name: str = "classical_repeat_purchase_rf_v1",
    db_path: str | Path = "data/retail.db",
) -> "PredictionTool | None":
    """Best-effort loader for build_agent(): returns None instead of raising
    so a missing/broken artifact disables the tool rather than crashing the
    whole agent at startup (same pattern build_agent already uses for the
    retriever and sql_tool)."""
    try:
        return PredictionTool.from_artifact(Path(models_dir) / artifact_name, db_path=db_path)
    except Exception as exc:
        print(f"[predict] prediction tool disabled: {exc}")
        return None
