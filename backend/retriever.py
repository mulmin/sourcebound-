"""검색 모듈: 하이브리드 회수(dense + BM25) → RRF 융합 → 리랭킹.

1) dense(임베딩 코사인)와 BM25(어휘)로 각각 후보를 회수하고,
2) RRF(Reciprocal Rank Fusion)로 두 순위를 융합해 1차 후보 풀을 만든 뒤,
3) 교차 인코더 리랭커가 있으면 "질문에 답이 되는가"로 재정렬해 최종 Top-k를 낸다.
리랭커/BM25가 없으면 각 단계가 자연스럽게 폴백한다(끝까지 도는 파이프라인 유지).

또한 답변 거부 판정을 위한 신호(dense/BM25/리랭커 최고점)를 함께 산출한다.
"""
from __future__ import annotations
import json
import numpy as np

from backend.config import INDEX_DIR, TOP_K, CANDIDATE_POOL, RRF_K
from backend.embeddings import EmbeddingBackend, cosine
from backend.lexical import Tokenizer, BM25Index
from backend.reranker import Reranker


def _rrf_fuse(rank_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """여러 순위 리스트를 RRF로 융합. {chunk_idx: 융합점수}."""
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class Retriever:
    def __init__(self):
        self.vectors = np.load(INDEX_DIR / "vectors.npy")
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.docs = json.loads((INDEX_DIR / "docs.json").read_text(encoding="utf-8"))
        texts = [c["text"] for c in self.chunks]
        # TF-IDF 폴백일 때는 코퍼스로 다시 fit해야 질의를 같은 공간에 사상 가능
        self.backend = EmbeddingBackend(corpus=texts)
        self.tokenizer = Tokenizer()
        self.bm25 = BM25Index(texts, self.tokenizer)
        self.reranker = Reranker()

    def search(self, query: str, k: int = TOP_K,
               exclude: set[str] | None = None) -> list[dict]:
        """exclude: 검색에서 제외할 doc_id 집합(소스 On/Off). 꺼진 출처의 청크는
        어떤 단계에도 후보로 올라오지 않는다(= 구독 해지 시 지식 비활성화)."""
        n = len(self.chunks)
        excl = exclude or set()
        allowed = [i for i in range(n) if self.chunks[i]["doc_id"] not in excl]
        if not allowed:
            self.last_signals = {"dense_top": 0.0, "bm25_top": 0.0,
                                 "rerank_top": None, "reranker_used": False}
            return []
        k = min(k, len(allowed))

        # 1) dense 회수 (허용된 청크만 순위에 올림)
        qv = self.backend.encode([query], kind="query")
        dense = cosine(qv, self.vectors)[0]
        dense_rank = sorted(allowed, key=lambda i: -dense[i])[:CANDIDATE_POOL]

        # 2) BM25 회수(가용 시)
        bm25 = self.bm25.scores(query)
        rank_lists = [dense_rank]
        if bm25 is not None:
            bm25_rank = sorted(allowed, key=lambda i: -bm25[i])[:CANDIDATE_POOL]
            rank_lists.append(bm25_rank)
        else:
            bm25 = np.zeros(n, dtype=np.float32)

        # 3) RRF 융합 → 1차 후보 풀
        fused = _rrf_fuse(rank_lists)
        pool = sorted(fused, key=lambda i: -fused[i])[:CANDIDATE_POOL]

        # 4) 리랭킹(가용 시) — 후보 풀을 교차 인코더로 재채점
        import time
        rerank_scores: dict[int, float] = {}
        rerank_s = 0.0
        if self.reranker.available:
            passages = [self.chunks[i]["text"] for i in pool]
            _t = time.monotonic()
            scored = self.reranker.score(query, passages)  # CPU에서 가장 비싼 단계
            rerank_s = time.monotonic() - _t
            for idx, sc in zip(pool, scored):
                rerank_scores[idx] = sc
            order = sorted(pool, key=lambda i: -rerank_scores[i])[:k]
        else:
            order = pool[:k]
        # 단계별 응답시간 진단용(리랭커가 몇 초/몇 쌍인지)
        self.last_timing = {"rerank_s": rerank_s, "rerank_pairs": len(pool)}

        results = []
        for i in order:
            c = dict(self.chunks[i])
            c["score"] = float(dense[i])           # 화면 표시용 '검색 유사도'(dense 코사인)
            c["bm25_score"] = float(bm25[i])
            c["rerank_score"] = rerank_scores.get(i)
            c["fused_score"] = float(fused.get(i, 0.0))
            c["doc_meta"] = self.docs[c["doc_id"]]["meta"]
            results.append(c)

        # 거부 판정용 신호 (허용된 청크 기준)
        self.last_signals = {
            "dense_top": max(float(dense[i]) for i in allowed),
            "bm25_top": max(float(bm25[i]) for i in allowed),
            "rerank_top": max(rerank_scores.values()) if rerank_scores else None,
            "reranker_used": self.reranker.available,
        }
        return results

    def is_refused(self, results: list[dict]) -> bool:
        """검색 신호로 '근거 없음' 판정.

        신호별 신뢰도가 달라 우선순위를 둔다:
          1) 리랭커 점수(가장 calibrated)가 있으면 그것으로 판정.
          2) sbert dense는 신뢰 가능 → dense 또는 BM25 중 하나라도 근거면 응답.
          3) TF-IDF(char n-gram) dense는 조사·어미로 false high가 나므로 신뢰 불가
             → BM25(내용어 기반)가 가용하면 BM25만으로 판정, 없으면 약한 폴백.
        """
        from backend.config import (
            SIM_REJECT_THRESHOLD, BM25_REJECT_THRESHOLD, RERANKER_REJECT_THRESHOLD,
        )
        if not results:
            return True
        sig = self.last_signals
        # 1) 리랭커(교차 인코더)가 가장 신뢰할 만한 관련성 신호
        if sig.get("rerank_top") is not None:
            return sig["rerank_top"] < RERANKER_REJECT_THRESHOLD
        # 2) BM25(내용어)는 임베딩 종류와 무관하게 OOD에 0을 주어 변별력이 좋다.
        #    (e5 dense는 무관 텍스트에도 ~0.79를 줘 절대값으로 OOD를 가르기 어렵다)
        if self.bm25.available:
            return not (sig["bm25_top"] > BM25_REJECT_THRESHOLD)
        # 3) 최후 폴백: dense 절대 임계(신뢰도 낮음)
        return not (sig["dense_top"] >= SIM_REJECT_THRESHOLD[self.backend.kind])

    def get_source(self, chunk_id: str) -> dict | None:
        for c in self.chunks:
            if c["chunk_id"] == chunk_id:
                doc = self.docs[c["doc_id"]]
                return {"chunk": c, "meta": doc["meta"], "body": doc["body"]}
        return None
