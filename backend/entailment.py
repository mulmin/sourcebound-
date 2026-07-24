"""함의/극성 검증 — 코사인 유사도가 못 잡는 '부정문 모순'을 겨냥한다.

코사인 유사도는 "비슷한가"를 잴 뿐 "근거가 문장을 함의하는가"를 보장하지
않는다. 특히 "먹여도 된다" vs "먹이면 안 된다"는 단어가 겹쳐 유사도가 높지만
의미는 정반대다(계획서 3.4(5)·6.6(2)의 한계).

3중 구조:
  1) polarity(): 허용/금지 극성을 표면 신호로 추정(무료·무모델).
  2) negation_conflict(): 답변 문장과 근거의 극성이 엇갈리면 '모순 의심'.
  3) nli(): 한국어 NLI 교차 인코더가 있으면 entail/contradict 판정(자동 활성화),
     없으면 None을 반환해 호출 측이 1)·2) 휴리스틱으로 폴백.
"""
from __future__ import annotations
import re

# 금지/부정 신호 (강 → 약)
NEG_MARKERS = [
    "안 된다", "안된다", "안 돼", "하면 안", "되지 않", "하지 않", "않는다",
    "말고", "말아야", "마세요", "금지", "금한다", "비권장", "권장되지", "권장하지",
    "불가", "삼가", "피한다", "피해야", "피하", "주의", "위험", "임의로 투여하지",
    "투여하지", "먹이지", "주지", "없다", "없이", "못한다", "불충분",
]
# 허용/긍정 신호
POS_MARKERS = [
    "해도 된다", "해도 됩니다", "할 수 있다", "투여할 수 있다", "사용할 수 있다",
    "가능하다", "권장한다", "권장된다", "권장되며", "권장", "괜찮다", "안전하다",
    "시작한다", "먹여도", "먹일 수", "주어도", "복용할 수",
]


def polarity(text: str) -> int:
    """+1=허용/긍정, -1=금지/부정, 0=중립. 신호 빈도 차로 추정."""
    neg = sum(text.count(m) for m in NEG_MARKERS)
    pos = sum(text.count(m) for m in POS_MARKERS)
    if neg > pos:
        return -1
    if pos > neg:
        return 1
    return 0


_tokenizer = None


def _content_tokens(text: str) -> set[str]:
    """비교용 내용어 토큰. kiwi 형태소(조사·어미 제거)가 있으면 사용해
    '마사지로' vs '마사지는'이 같은 '마사지'로 묶이게 한다. 없으면 정규식 폴백."""
    global _tokenizer
    if _tokenizer is None:
        from backend.lexical import Tokenizer
        _tokenizer = Tokenizer()
    toks = {t for t in _tokenizer.tokenize(text) if len(t) >= 2}
    return toks or {w for w in re.findall(r"[A-Za-z0-9가-힣]{2,}", text)}


def _split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.다요])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 4]


def negation_conflict(sentence: str, evidence_text: str,
                      min_overlap: int = 2) -> bool:
    """답변 문장과 근거의 극성이 반대이고 내용어가 겹치면 '모순 의심'.

    근거 청크는 흔히 여러 주장을 함께 담으므로(예: '6개월 이상 가능' + '6개월
    미만 금지'), 청크 전체 극성과 비교하면 오탐이 난다. 따라서 답변 문장과
    내용어가 가장 많이 겹치는 '청크 내 한 문장'을 찾아 그 극성과 비교한다."""
    ps = polarity(sentence)
    if ps == 0:
        return False
    stoks = _content_tokens(sentence)
    best, best_ov = None, 0
    for es in _split(evidence_text):
        ov = len(stoks & _content_tokens(es))
        if ov > best_ov:
            best_ov, best = ov, es
    if best is None or best_ov < min_overlap:
        return False
    pe = polarity(best)
    return pe != 0 and pe != ps


# ---- 선택적 NLI 층 (모델 있으면 자동 활성화) ----
_nli_model = None
_nli_tried = False


def _get_nli():
    global _nli_model, _nli_tried
    if _nli_tried:
        return _nli_model
    _nli_tried = True
    try:
        from sentence_transformers import CrossEncoder
        # 다국어 NLI 교차 인코더(설치/다운로드되어 있을 때만)
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    except Exception:
        _nli_model = None
    return _nli_model


def nli(premise: str, hypothesis: str) -> str | None:
    """premise(근거)가 hypothesis(답변문장)를 함의하는지.

    반환: 'entailment' | 'neutral' | 'contradiction' | None(모델 없음)."""
    model = _get_nli()
    if model is None:
        return None
    scores = model.predict([(premise, hypothesis)])[0]
    labels = ["contradiction", "entailment", "neutral"]  # 모델 라벨 순서
    return labels[int(scores.argmax())]
