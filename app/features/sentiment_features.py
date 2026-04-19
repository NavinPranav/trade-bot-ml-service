"""Sentiment features using FinBERT."""
from typing import List, Dict
from loguru import logger


class SentimentAnalyzer:
    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = "ProsusAI/finbert"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")

    def analyze(self, texts: List[str]) -> List[Dict]:
        self._load_model()
        if self._model is None:
            return [{"sentiment": "neutral", "score": 0.0} for _ in texts]

        import torch

        results = []
        for text in texts:
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]

            labels = ["positive", "negative", "neutral"]
            idx = probs.argmax().item()
            # Score: positive = +1, negative = -1, neutral = 0
            score = probs[0].item() - probs[1].item()

            results.append({"sentiment": labels[idx], "score": round(score, 4)})

        return results

    def get_aggregate_sentiment(self, headlines: List[str]) -> float:
        if not headlines:
            return 0.0
        results = self.analyze(headlines)
        return sum(r["score"] for r in results) / len(results)


# Singleton
sentiment_analyzer = SentimentAnalyzer()