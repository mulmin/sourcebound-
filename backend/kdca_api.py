"""질병관리청 국가건강정보포털 오픈API 클라이언트.

data.go.kr(15087442)에서 발급한 TOKEN으로 api.kdca.go.kr를 호출한다.
- healthInfoList  : 전체 건강정보 기사 목록(제목 + cntntsSn) 약 680건. 서버측 검색 없음.
- healthInfo(sn)  : 특정 기사 본문(섹션별 CNTNTS_CL_NM/CNTNTS_CL_CN).

정부 서버가 구형 TLS 재협상을 써서 OpenSSL 3.x가 막으므로 레거시 옵션을 켠다.
TOKEN은 프로젝트 루트 .env 의 DATA_GO_KR_KEY 에서 읽는다(코드/로그에 노출 금지).
"""
from __future__ import annotations
import ssl
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

from backend.config import ROOT

BASE = "http://api.kdca.go.kr/api/provide"


def _load_token() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "DATA_GO_KR_KEY":
            return v.strip().strip('"').strip("'")
    raise RuntimeError(".env 에 DATA_GO_KR_KEY 가 없습니다")


def _ctx() -> ssl.SSLContext:
    c = ssl.create_default_context()
    c.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


_CDATA = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.S)


class KdcaClient:
    def __init__(self, token: str | None = None, delay: float = 0.3):
        self.token = token or _load_token()
        self.ctx = _ctx()
        self.delay = delay

    def _get(self, path: str, **params) -> str:
        q = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        url = f"{BASE}/{path}?TOKEN={urllib.parse.quote(self.token)}" + (("&" + q) if q else "")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30, context=self.ctx) as r:
            body = r.read().decode("utf-8", "ignore")
        if self.delay:
            time.sleep(self.delay)
        return body

    def list_articles(self) -> list[dict]:
        """전체 기사 목록 [{'sn','title'}]. (서버 검색 없음 — 전체 반환)"""
        xml = self._get("healthInfoList")
        items = []
        for block in re.findall(r"<cntntsList>(.*?)</cntntsList>", xml, re.S):
            sj = _CDATA.search(re.search(r"<CNTNTS_SJ>.*?</CNTNTS_SJ>", block, re.S).group(0)) \
                if "<CNTNTS_SJ>" in block else None
            sn = re.search(r"<CNTNTS_SN><!\[CDATA\[(\d+)\]\]>", block)
            if sj and sn:
                items.append({"sn": sn.group(1), "title": sj.group(1).strip()})
        return items

    def article(self, sn: str) -> dict:
        """기사 본문. {'sn','title','sections':[(name,text)], 'body'}"""
        xml = self._get("healthInfo", cntntsSn=sn)
        title_m = re.search(r"<CNTNTS_SJ><!\[CDATA\[(.*?)\]\]>", xml)
        title = title_m.group(1).strip() if title_m else ""
        # 답변 근거로 가치 없는 섹션(서지정보·키워드 나열)은 제외
        SKIP = {"참고문헌", "연관 주제어", "연관주제어"}
        sections = []
        for block in re.findall(r"<cntntsCl>(.*?)</cntntsCl>", xml, re.S):
            nm = re.search(r"<CNTNTS_CL_NM><!\[CDATA\[(.*?)\]\]>", block, re.S)
            cn = re.search(r"<CNTNTS_CL_CN><!\[CDATA\[(.*?)\]\]>", block, re.S)
            if nm and cn:
                name = nm.group(1).strip()
                if name in SKIP:
                    continue
                text = re.sub(r"\s+", " ", cn.group(1)).strip()
                if text:
                    sections.append((name, text))
        body = "\n\n".join(f"{n}\n{t}" for n, t in sections)
        return {"sn": sn, "title": title, "sections": sections, "body": body}
