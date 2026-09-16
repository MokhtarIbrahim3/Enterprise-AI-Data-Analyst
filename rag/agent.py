
class AnalyticsAgent:

    def __init__(
        self,
        llm,
        retriever,
        sql_tool
    ):
        self.llm = llm
        self.retriever = retriever
        self.sql_tool = sql_tool

    def run(self, question):

        # 1. Understand question
        # 2. Decide tool
        # 3. Execute tool
        # 4. Generate grounded answer
        # 5. Return sources

        ...
