"""라이선스 분류와 정책 — "이용 조건 스티커"를 읽는 규칙.

지식팩 문서마다 license 메타데이터가 제각각이다(공공누리 유형, All rights
reserved, 미상 …). 이 모듈은 그 문자열을 4등급으로 분류하고, 등급별로
시스템이 지켜야 할 행동(재서술 허용 여부)을 정한다.

등급:
  open    : 출처표시만 지키면 자유 이용 → 청킹·재서술(LLM) 허용
  nd      : 변경금지(공공누리 제4유형 등) → 원문 문장 그대로만 인용
  arr     : All rights reserved → 데모 한정, 원문 그대로만 인용
  unknown : 이용조건 미확인 → 보수적으로 원문 그대로만 인용
"""
from __future__ import annotations


def lic_class(license_str: str | None) -> str:
    s = (license_str or "").strip()
    if not s or s.lower() == "unknown":
        return "unknown"
    low = s.lower()
    if "all rights" in low:
        return "arr"
    if "변경금지" in s or "-nd" in low:
        return "nd"
    if "공공누리" in s or "출처표시" in s or "cc" in low:
        return "open"
    return "unknown"


def allow_rewrite(cls: str) -> bool:
    """LLM이 이 근거를 재서술(요약·의역)해도 되는가. open만 허용."""
    return cls == "open"


LABELS = {
    "open": "자유이용(출처표시)",
    "nd": "변경금지 · 원문 인용만",
    "arr": "All rights reserved · 원문 인용만",
    "unknown": "이용조건 미확인 · 원문 인용만",
}
