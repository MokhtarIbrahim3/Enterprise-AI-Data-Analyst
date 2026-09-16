"""
Prediction tool: wraps the classical repeat-purchase model artifact so the
agent can call it as a third tool alongside SQL and RAG.

Design choice: rather than waiting on a hand-off ("send me the function
signature"), this loads models/<name>.joblib + models/<name>.json and
introspects the metadata json for the feature list, target name and
reported metrics. If a team member changes the model later, as long as the
.json sidecar has a "features": [...] key (or the joblib model exposes
sklearn's feature_names_in_), this keeps working with no code changes here.

    tool = PredictionTool.from_artifact("models/classical_repeat_purchase_rf_v1")
    tool.predict(features={"recency_days": 12, "frequency": 4, ...})
    tool.predict(customer_id="12583")   # looks the row up in the DB instead

Drop this file at: src/models/predict.py
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


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
            f"classical model (version {self.model_version}). "
            f"Required features: {', '.join(self.features)}. "
            f"Reported offline metrics: {metric_str}. "
            "Use this tool when the question asks to predict, forecast, or "
            "estimate the likelihood of a customer's FUTURE behavior "
            "(e.g. repeat purchase, churn) rather than report a historical "
            "fact (use sql) or a definition (use rag)."
        )

    # ------------------------------------------------------------------ #
    def _fetch_features_for_customer(self, customer_id: str) -> dict[str, Any]:
        """Pulls a feature row from the customer_features table so the tool
        can be called from natural language with just a customer id.
        Assumes a customer_features table/view exists in the SQLite db with
        at minimum a customer_id column plus this model's feature columns
        (this mirrors the spark/customer_features feature-store output)."""
        if not self.db_path or not Path(self.db_path).exists():
            raise RuntimeError("No database configured to look up customer features.")
        cols = ", ".join(self.features)
        query = f"SELECT {cols} FROM customer_features WHERE customer_id = ?"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(query, (customer_id,)).fetchone()
        if row is None:
            raise KeyError(f"No feature row found for customer_id={customer_id!r}")
        return {k: row[k] for k in self.features}

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

            row = [[features[f] for f in self.features]]
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
