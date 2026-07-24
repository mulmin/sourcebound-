"""정량 성능 평가 (수행계획서 4장).

동일 평가 세트로 다음을 측정한다:
  (1) 검색 성능   : Recall@k, MRR@k         (양성 질의)
  (2) 답변 거부   : 거부 재현율, 오거부율     (음성/양성 질의)
  (3) 인용 충실도 : 평균 답변-근거 유사도, 모순율, 미인용율  (응답한 양성 질의)
       ※ 인용 정확도(Citation Precision)·환각률의 '엄밀한' 판정은 사람/NLI 검수가
          필요하므로, 여기서는 자동 산출 가능한 '근사 지표(proxy)'를 함께 제시한다.

사용: python -m eval.run_eval  [k]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from backend.config import SIM_WARN_THRESHOLD
from backend.retriever import Retriever
from backend.generator import generate
from backend.verifier import verify


def load_questions():
    path = Path(__file__).parent / "questions.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(k: int = 5):
    qs = load_questions()
    pos = [q for q in qs if not q.get("is_negative")]
    neg = [q for q in qs if q.get("is_negative")]
    r = Retriever()
    print(f"임베딩={r.backend.kind}  BM25={r.bm25.available}  리랭커={r.reranker.available}")
    print(f"질의: 양성 {len(pos)} / 음성 {len(neg)}   (k={k})")
    print("=" * 74)

    # --- (1) 검색 성능 + (2) 오거부율: 양성 질의 ---
    recall_hit = 0
    mrr_sum = 0.0
    false_refusal = 0
    sims_all, n_grounded, n_contra, n_uncited, n_stmt = [], 0, 0, 0, 0
    answered = 0
    for q in pos:
        ev = r.search(q["question"], k=k)
        docs = [c["doc_id"] for c in ev]
        # Recall@k / MRR@k — 복수 정답 지원(relevant_docs 리스트, 구버전 relevant_doc 호환).
        # 코퍼스가 커지면 같은 주제를 여러 문서가 다루므로 '어느 정답이든 회수'가 기준.
        rel = set(q.get("relevant_docs") or
                  ([q["relevant_doc"]] if q.get("relevant_doc") else []))
        hit_ranks = [docs.index(d) for d in rel if d in docs]
        if hit_ranks:
            recall_hit += 1
            mrr_sum += 1.0 / (min(hit_ranks) + 1)
        # 오거부(응답해야 하는데 거부)
        if r.is_refused(ev):
            false_refusal += 1
            continue
        answered += 1
        # 인용 충실도(응답한 경우만)
        gen = generate(q["question"], ev, backend=r.backend)
        for v in verify(gen["sentences"], ev, backend=r.backend):
            if v["status"] == "advisory":
                continue
            n_stmt += 1
            if v["status"] in ("grounded", "contradiction"):
                n_grounded += 1
                sims_all.append(v["similarity"])
                if v["contradiction"]:
                    n_contra += 1
            elif v["status"] == "uncited":
                n_uncited += 1

    # --- (2) 거부 재현율: 음성 질의 ---
    refused_ok = sum(1 for q in neg if r.is_refused(r.search(q["question"], k=k)))

    P = len(pos) or 1
    N = len(neg) or 1
    G = n_grounded or 1
    S = n_stmt or 1
    print("(1) 검색 성능")
    print(f"    Recall@{k}      = {recall_hit}/{len(pos)} = {recall_hit/P:.2f}   (목표 ≥ 0.8)")
    print(f"    MRR@{k}         = {mrr_sum/P:.3f}")
    print("(2) 답변 거부 적정성")
    print(f"    거부 재현율     = {refused_ok}/{len(neg)} = {refused_ok/N:.2f}   (목표 ≥ 0.8)")
    print(f"    오거부율        = {false_refusal}/{len(pos)} = {false_refusal/P:.2f}   (목표 ≤ 0.1)")
    print("(3) 인용 충실도 (응답한 양성 질의 기준, 근사 지표)")
    print(f"    응답 문장 수    = {n_stmt} (근거인용 {n_grounded}, 미인용 {n_uncited})")
    print(f"    평균 답변-근거 유사도 = {sum(sims_all)/G:.3f}")
    print(f"    근거 약함(<{SIM_WARN_THRESHOLD}) 비율 = {sum(1 for s in sims_all if s < SIM_WARN_THRESHOLD)}/{n_grounded}")
    print(f"    모순 의심 비율  = {n_contra}/{n_grounded}")
    print(f"    미인용(잠재 환각) 비율 = {n_uncited}/{n_stmt} = {n_uncited/S:.2f}")
    print("=" * 74)
    print("※ 인용 정확도·환각률의 엄밀 판정은 사람/NLI 검수 필요. RAG on/off 환각 비교는")
    print("  순수 LLM 베이스라인(ANTHROPIC_API_KEY)이 있어야 측정 가능(확장).")


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(k)
