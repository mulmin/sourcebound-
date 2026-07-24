"""지식 팩 구축 파이프라인 (오프라인).

data/knowledge_pack/ 의 .md / .pdf / .hwp / .hwpx 문서를 읽어 청크 분할·임베딩 후
data/index/에 저장한다. 각 청크는 원문 내 오프셋(start, end)을 보존하여 원문 뷰어의
구절 하이라이트에 사용한다.

문서 형식:
  - .md  : 머리에 `키: 값` 메타데이터 블록, `---` 구분선, 이후 본문.
  - 그 외: 텍스트를 자동 추출하고, 메타데이터는 사이드카 파일 `<파일명>.meta`
           (같은 `키: 값` 형식)에서 읽는다. 없으면 파일명을 제목으로 쓴다.
메타데이터 권장 키: title, publisher, year, url, license  (출처 표기·원문 링크·라이선스 인지용)
"""
from __future__ import annotations
import json
import re
import numpy as np

from backend.config import (
    KNOWLEDGE_PACK_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
)
from backend.embeddings import EmbeddingBackend

SUPPORTED_EXTS = (".md", ".pdf", ".hwp", ".hwpx")


def _parse_meta_lines(text: str) -> dict:
    meta = {}
    for line in text.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


# 단독 다운로드/파일 링크 줄(본문 가치 없음). 문장 속 참고 URL은 살린다.
_JUNK_LINE = re.compile(
    r"^\s*(?:https?://\S+|\S*(?:fileDown|FileDown|healthInfoFileDown)\S*|\S*\?SEQ=\S*)\s*$")


def _clean_extracted(text: str) -> str:
    """추출 텍스트 정리: 제어문자, 페이지번호 줄, 단독 다운로드 링크, 과도한 빈 줄 제거."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)  # PDF 제어문자
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.fullmatch(r"[-–—\s]*\d{1,3}[-–—\s]*", s):   # '- 12 -' / '35' 페이지번호
            continue
        if _JUNK_LINE.match(s):                            # 단독 다운로드 링크
            continue
        lines.append(ln.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_pdf(path) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    try:
        return "\n\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _extract_hwpx(path) -> str:
    """HWPX = zip + XML. 본문 XML에서 태그를 제거해 텍스트만 추출."""
    import zipfile
    parts = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.startswith("Contents/") and name.endswith(".xml"):
                xml = z.read(name).decode("utf-8", "ignore")
                xml = re.sub(r"<[^>]+>", " ", xml)  # 태그 제거
                parts.append(xml)
    return "\n".join(parts)


def _extract_hwp(path) -> str:
    """레거시 HWP(바이너리)는 pyhwp의 hwp5txt CLI로 추출(설치 시)."""
    import subprocess
    try:
        out = subprocess.run(["hwp5txt", str(path)], capture_output=True, timeout=60)
        return out.stdout.decode("utf-8", "ignore")
    except Exception as e:
        raise RuntimeError(f"HWP 추출 실패({path.name}): hwp5txt 필요 — {e}")


def parse_document(path) -> dict:
    ext = path.suffix.lower()
    if ext == ".md":
        raw = path.read_text(encoding="utf-8")
        meta, body = {}, raw
        if "\n---\n" in raw:
            head, body = raw.split("\n---\n", 1)
            meta = _parse_meta_lines(head)
        return {"id": path.stem, "meta": meta, "body": _clean_extracted(body)}

    # 비-md: 사이드카 메타 + 형식별 텍스트 추출
    side = path.with_name(path.name + ".meta")
    meta = _parse_meta_lines(side.read_text(encoding="utf-8")) if side.exists() else {}
    meta.setdefault("title", path.stem)
    if ext == ".pdf":
        body = _extract_pdf(path)
    elif ext == ".hwpx":
        body = _extract_hwpx(path)
    elif ext == ".hwp":
        body = _extract_hwp(path)
    else:
        raise ValueError(f"지원하지 않는 형식: {path.name}")
    return {"id": path.stem, "meta": meta, "body": _clean_extracted(body)}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """문단 경계를 우선 존중하며 size자 내외로 분할. (start, end, text) 반환."""
    chunks = []
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + size, n)
        if end < n:
            # 가까운 문단/문장 경계에서 끊기
            window = text[pos:end]
            cut = max(window.rfind("\n\n"), window.rfind("다."), window.rfind(". "))
            if cut > size // 2:
                end = pos + cut + 2
            else:
                # 창 안에 경계가 없으면 뒤로 조금 확장해 문장을 중간에서 자르지 않는다
                m = re.search(r"(?:다\.|요\.|\.)\s|\n\s*\n", text[end:end + size])
                if m:
                    end = end + m.end()
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append((pos, end, chunk))
        if end >= n:
            break
        # 다음 청크가 '문장 중간'에서 시작하지 않도록 겹침 구간 안에서 문장 경계로 스냅.
        # (그렇지 않으면 '떤 방법으로…', '작되어 1개월까지…' 같은 토막이 근거로 잡힌다)
        nxt = max(end - overlap, pos + 1)
        m = re.search(r"(?:다\.|요\.|\.)\s|\n\s*\n", text[nxt:end])
        pos = (nxt + m.end()) if m else end
    return chunks


def build_index() -> dict:
    paths = sorted(p for p in KNOWLEDGE_PACK_DIR.iterdir()
                   if p.suffix.lower() in SUPPORTED_EXTS)
    docs, skipped = [], []
    for p in paths:
        try:
            docs.append(parse_document(p))
        except Exception as e:            # 추출 실패 문서는 건너뛰고 계속
            skipped.append((p.name, str(e)))
            print(f"  건너뜀: {p.name} — {e}")
    if not docs:
        raise SystemExit(f"지식 팩 문서가 없습니다: {KNOWLEDGE_PACK_DIR}")

    records = []
    for doc in docs:
        for ci, (start, end, text) in enumerate(chunk_text(doc["body"])):
            records.append({
                "chunk_id": f"{doc['id']}--{ci}",
                "doc_id": doc["id"],
                "start": start,
                "end": end,
                "text": text,
            })

    texts = [r["text"] for r in records]
    backend = EmbeddingBackend(corpus=texts)
    vectors = backend.encode(texts)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    (INDEX_DIR / "docs.json").write_text(
        json.dumps({d["id"]: {"meta": d["meta"], "body": d["body"]} for d in docs},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (INDEX_DIR / "backend.json").write_text(
        json.dumps({"kind": backend.kind}), encoding="utf-8")
    return {"docs": len(docs), "chunks": len(records), "backend": backend.kind}


if __name__ == "__main__":
    info = build_index()
    print(f"인덱스 구축 완료: 문서 {info['docs']}건, 청크 {info['chunks']}개, "
          f"임베딩={info['backend']}")
