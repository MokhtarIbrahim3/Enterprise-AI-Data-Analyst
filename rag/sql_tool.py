
from .security import validate_sql


class SQLAnalyticsTool:

    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql):

        allowed, reason = validate_sql(sql)

        if not allowed:
            return {
                "status": "blocked",
                "reason": reason
            }

        result = self.connection.execute(sql)

        return {
            "status": "success",
            "data": result
        }
