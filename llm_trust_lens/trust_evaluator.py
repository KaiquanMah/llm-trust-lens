from .embedding_model_interface import EmbeddingModelInterface

class TrustEvaluator:
    def __init__(self, model: EmbeddingModelInterface):
        self.model = model

    def evaluate(self, trust_metrics: list[str]):
        # Implementation of evaluation logic
        pass

    def plot_results(self, metric: str):
        # Implementation of plotting logic
        pass
