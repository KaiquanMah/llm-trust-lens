from llm_trust_lens import TrustEvaluator
from llm_trust_lens.embedding_model_interface import EmbeddingModelInterface

class MyEmbeddingModel(EmbeddingModelInterface):
    def get_embedding(self, text: str):
        return some_embedding_function(text)  # User's implementation

    def retrieve_similar(self, query: str, top_k: int = 5):
        return some_retrieval_function(query, top_k)  # User's implementation

my_model = MyEmbeddingModel()
evaluator = TrustEvaluator(model=my_model)

report = evaluator.evaluate(trust_metrics=["bias", "retrieval", "hallucination"])
evaluator.plot_results()
