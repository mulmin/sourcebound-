"""출처 검증 모듈: 답변 문장 ↔ 인용 근거 청크 일치 점검.

생성 모듈이 넘겨준 '구조화된 문장 목록'을 입력으로 받으므로, 더 이상
정규식으로 문장을 추측 분할하지 않는다(인용-문장 매핑이 생성 의도대로 보존됨).

문장 분류:
  - grounded : 인용이 달린 진술 → 유사도 재계산 + 부정문 모순 점검(+NLI 가용 시)
  - advisory : 인용 없는 상담 권고 문장 → 검증 대상 아님(경고 노이즈 방지)
  - uncited  : 인용 없는 진술 → 근거 없는 주장일 수 있어 경고

검증 깊이(겹쳐서 적용):
  1) 코사인 유사도 → 화면 배지 %, 임계 미만이면 '근거 약함'.
  2) 부정어 극성 모순(무료) → '먹여도 된다 vs 먹이면 안 된다' 같은 반대 결론 탐지.
  3) NLI 함의 판정(모델 가용 시 자동) → entail/contradict 라벨.
"""
from __future__ import annotations

from backend.config import SIM_WARN_THRESHOLD
from backend.embeddings import EmbeddingBackend, cosine
from backend.entailment import negation_conflict, nli, polarity, _split, _content_tokens

_ADVISORY_HINTS = ("상담", "전문의", "의료기관", "병원")


def _is_advisory(text: str) -> bool:
    return any(h in text for h in _ADVISORY_HINTS)


def verify(sentences: list[dict], evidence: list[dict], backend=None) -> list[dict]:
    """문장별 검증 결과 리스트.

    sentences: [{"text", "citations": [n, ...]}]  (생성 모듈 출력)
    반환: [{"sentence","citations","similarity","warn","status",
            "contradiction","entailment"}]
    """
    ev_texts = [c["text"] for c in evidence]
    if not ev_texts or not sentences:
        return []
    if backend is None:
        clean_texts = [s["text"] for s in sentences]
        backend = EmbeddingBackend(corpus=ev_texts + clean_texts)
    ev_vecs = backend.encode(ev_texts)

    results = []
    for s in sentences:
        text = s["text"].strip()
        cites = [c for c in s.get("citations", []) if 1 <= c <= len(evidence)]
        if len(text) < 8:
            continue

        if cites:
            sv = backend.encode([text], kind="query")
            valid = [c - 1 for c in cites]
            sim = float(max(cosine(sv, ev_vecs[valid])[0]))

            # 답변 문장이 인용 청크에 그대로(verbatim) 들어있으면 직접 인용이므로
            # 모순일 수 없다(추출모드의 형제 문장 오매칭 방지).
            squashed = "".join(text.split())
            verbatim = any("".join(ev_texts[j].split()).find(squashed) != -1
                           for j in valid)

            # 부정문 모순 점검(무료) + NLI(가용 시) — 인용 근거들 중 하나라도 모순이면 경고
            contradiction = False
            entail_label = None
            if not verbatim:
                for j in valid:
                    if negation_conflict(text, ev_texts[j]):
                        contradiction = True
                    label = nli(ev_texts[j], text)  # premise=근거, hypothesis=답변문장
                    if label == "contradiction":
                        contradiction = True
                        entail_label = "contradiction"
                    elif label == "entailment" and entail_label is None:
                        entail_label = "entailment"

            results.append({
                "sentence": text, "citations": cites,
                "similarity": round(sim, 3),
                "warn": sim < SIM_WARN_THRESHOLD or contradiction,
                "status": "contradiction" if contradiction else "grounded",
                "contradiction": contradiction,
                "entailment": entail_label,
            })
        elif _is_advisory(text):
            results.append({
                "sentence": text, "citations": [], "similarity": None,
                "warn": False, "status": "advisory",
                "contradiction": False, "entailment": None,
            })
        else:
            results.append({
                "sentence": text, "citations": [], "similarity": None,
                "warn": True, "status": "uncited",
                "contradiction": False, "entailment": None,
            })
    return results


def detect_source_conflict(evidence: list[dict], min_overlap: int = 3) -> dict | None:
    """검색된 근거들 사이의 '출처 간 상충' 탐지.

    서로 다른 문서의 '같은 주장'(내용어를 충분히 공유하는 문장 쌍)이 극성이
    반대이면 충돌로 본다. 문장 쌍 단위로 비교하고 공유 내용어 수(min_overlap)를
    크게 요구해, 주제가 다른 청크를 충돌로 오판하지 않는다.
    충돌 시 양쪽을 모두 인용해 사용자 판단에 위임한다(계획서 3.4(3)·6.6(3))."""
    for i in range(len(evidence)):
        for j in range(i + 1, len(evidence)):
            a, b = evidence[i], evidence[j]
            if a["doc_id"] == b["doc_id"]:
                continue
            for sa in _split(a["text"]):
                pa = polarity(sa)
                if pa == 0:
                    continue
                ta = _content_tokens(sa)
                for sb in _split(b["text"]):
                    pb = polarity(sb)
                    if pb == 0 or pb == pa:
                        continue
                    if len(ta & _content_tokens(sb)) >= min_overlap:
                        return {"between": [i + 1, j + 1],
                                "docs": [a["doc_id"], b["doc_id"]],
                                "claims": [sa, sb]}
    return None
