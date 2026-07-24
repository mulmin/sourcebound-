"""임베딩 백엔드.

sentence-transformers가 설치되어 있으면 다국어 임베딩 모델을 사용하고,
없으면 TF-IDF(문자 n-gram)로 폴백한다. 폴백 모드 덕분에 의존성 설치
없이도 전체 파이프라인이 동작한다(데모/CI용).
"""
from __future__ import annotations
import numpy as np


class EmbeddingBackend:
    def __init__(self, corpus: list[str] | None = None):
        self.kind = "tfidf"
        self._model = None
        self._vectorizer = None
        self._name = ""
        try:
            from sentence_transformers import SentenceTransformer
            from backend.config import EMBEDDING_MODEL
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            self._name = EMBEDDING_MODEL
            self.kind = "sbert"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), max_features=50000
            )
            if corpus:
                self._vectorizer.fit(corpus)

    def fit(self, corpus: list[str]) -> None:
        if self.kind == "tfidf":
            self._vectorizer.fit(corpus)

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        """kind: 'query' | 'passage'. e5 계열 모델은 비대칭 프리픽스를 요구하므로
        질의에는 'query: ', 문서에는 'passage: '를 붙여야 유사도가 제대로 나온다.
        e5가 아닌 모델/TF-IDF에서는 프리픽스가 무시된다."""
        if self.kind == "sbert":
            inputs = texts
            if "e5" in self._name.lower():
                prefix = "query: " if kind == "query" else "passage: "
                inputs = [prefix + t for t in texts]
            vecs = self._model.encode(inputs, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        vecs = self._vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """정규화된 벡터 간 코사인 유사도 행렬."""
    return a @ b.T
