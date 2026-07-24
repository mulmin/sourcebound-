"""국가건강정보포털 오픈API에서 영유아·소아 관련 건강정보를 수집해 지식팩으로 저장.

실행: python -m backend.kdca_collect
전체 기사(약 680건)를 훑어 소아 관련 마커가 충분한 글만 골라 data/knowledge_pack/에
kdca_*.md 로 저장한다(출처·라이선스 메타 포함). 서버측 검색이 없어 내용 기반으로 선별.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

from backend.config import KNOWLEDGE_PACK_DIR
from backend.kdca_api import KdcaClient

# 소아 도메인 판별: '강한 마커'가 여러 종류 나오고, 글 전체에서 밀도도 충분해야 한다.
# (단순 등장 횟수만 보면 성인 질환 글이 '소아'를 몇 번 언급했다는 이유로 섞여 들어와
#  "성인 우울증 약" 같은 도메인 밖 질문에 답해버리는 문제가 생긴다 — 실측으로 확인됨)
STRONG = ["영유아", "소아", "신생아", "영아", "유아기", "어린이",
          "이유식", "모유", "예방접종", "수유"]
WEAK = ["아기", "생후", "개월", "돌 전"]
MIN_DISTINCT = 2     # 서로 다른 강한 마커 종류 수
MIN_DENSITY = 1.2    # 1,000자당 마커 등장 수
MIN_LEN = 400        # 본문 최소 길이(자)
MAX_KEEP = 300       # 상한
LICENSE = "공공데이터포털 오픈API · 출처표시 (질병관리청 국가건강정보포털)"
PORTAL = "https://health.kdca.go.kr/"


def _safe(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:40]


def pediatric_stats(text: str) -> tuple[int, float]:
    """(강한 마커 고유 종류 수, 1000자당 마커 밀도)"""
    n = len(text) or 1
    distinct = sum(1 for m in STRONG if m in text)
    total = sum(text.count(m) for m in STRONG) + sum(text.count(m) for m in WEAK)
    return distinct, total / n * 1000


def is_pediatric(text: str) -> bool:
    d, den = pediatric_stats(text)
    return d >= MIN_DISTINCT and den >= MIN_DENSITY


def main():
    c = KdcaClient(delay=0.1)
    lst = c.list_articles()
    print(f"전체 목록: {len(lst)}건 — 본문 수집·선별 시작", flush=True)

    kept = []
    for i, a in enumerate(lst):
        try:
            art = c.article(a["sn"])
        except Exception as e:
            continue
        body = art["body"]
        if len(body) < MIN_LEN:
            continue
        if is_pediatric(body):
            d, den = pediatric_stats(body)
            kept.append({"sn": a["sn"], "title": a["title"], "body": body,
                         "score": round(den, 2)})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(lst)} 처리, 선별 {len(kept)}건", flush=True)

    kept.sort(key=lambda x: -x["score"])
    kept = kept[:MAX_KEEP]

    # 기존 kdca_*.md 정리 후 재작성
    for old in KNOWLEDGE_PACK_DIR.glob("kdca_*.md"):
        old.unlink()
    for a in kept:
        stem = f"kdca_{a['sn']}_{_safe(a['title'])}"
        md = (f"title: {a['title']}\n"
              f"publisher: 질병관리청 국가건강정보포털\n"
              f"url: {PORTAL}\n"
              f"license: {LICENSE}\n"
              f"---\n{a['body']}\n")
        (KNOWLEDGE_PACK_DIR / f"{stem}.md").write_text(md, encoding="utf-8")

    print(f"\n수집 완료: {len(kept)}건 저장 (점수 {kept[-1]['score']}~{kept[0]['score']})")
    print("상위 10:", [a["title"][:18] for a in kept[:10]])


if __name__ == "__main__":
    main()
