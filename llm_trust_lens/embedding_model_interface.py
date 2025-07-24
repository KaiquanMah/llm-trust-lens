class EmbeddingModelInterface:
    def get_embedding(self, text: str) -> list[float]:
        """Returns an embedding vector for the input text."""
        raise NotImplementedError

    def retrieve_similar(self, query: str, top_k: int = 5) -> list[str]:
        """Returns top-k retrieved documents for a given query."""
        raise NotImplementedError
