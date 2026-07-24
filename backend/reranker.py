"""리랭커: 교차 인코더(cross-encoder)로 (질문, 청크) 쌍을 함께 채점.

bi-encoder 임베딩 유사도는 '주제 유사성'을 잴 뿐 '질문에 답이 되는가(관련성)'를
직접 보장하지 않는다. 교차 인코더는 질문과 청크를 한 입력으로 함께 인코딩해
실제 관련성을 더 정밀하게 채점한다(계획서 3.4(2)·6.6(1) 대응).

sentence-transformers의 CrossEncoder가 설치돼 있으면 사용하고, 없으면
available=False로 두어 호출 측이 융합 점수로 폴백하게 한다.
"""
from __future__ import annotations


class Reranker:
    def __init__(self):
        self._model = None
        self.available = False
        # 환경변수 MPA_DISABLE_RERANKER=1 이면 무거운 리랭커 로드를 건너뛴다
        # (기동 속도 우선 시). 이 경우 검색은 하이브리드 융합 점수로 폴백한다.
        import os
        if os.environ.get("MPA_DISABLE_RERANKER") == "1":
            return
        try:
            from sentence_transformers import CrossEncoder
            from backend.config import RERANKER_MODEL
            self._model = CrossEncoder(RERANKER_MODEL)
            self.available = True
        except Exception:
            self.available = False

    def score(self, query: str, passages: list[str]) -> list[float]:
        """각 passage의 관련성 점수(로짓). 미가용 시 빈 리스트."""
        if not self.available or not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]
