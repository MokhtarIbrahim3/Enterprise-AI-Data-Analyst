
class Retriever:

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, k=5):

        results = self.vector_store.search(
            query,
            k=k
        )

        return results
