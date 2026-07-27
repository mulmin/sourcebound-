"""FastAPI 서버: 챗봇 API + 프론트엔드 정적 서빙.

실행:  uvicorn backend.main:app --reload
접속:  http://localhost:8000
"""
from __future__ import annotations
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import ROOT, TOP_K
from backend.retriever import Retriever
from backend.generator import generate
from backend.verifier import verify, detect_source_conflict
from backend.entailment import polarity, _split, _content_tokens
from backend.licensing import lic_class, LABELS

app = FastAPI(title="My Private AI — 출처 기반 육아 지식 챗봇")
_retriever: Retriever | None = None

# ---- 런타임 상태: 꺼진 출처(소스 On/Off) + 출처별 인용 횟수(기여도) ----
import json as _json
STATE_PATH = ROOT / "data" / "runtime_state.json"


def _load_state():
    try:
        s = _json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return set(s.get("disabled", [])), dict(s.get("citations", {}))
    except Exception:
        return set(), {}


_disabled, _citations = _load_state()


def _save_state():
    try:
        STATE_PATH.write_text(_json.dumps(
            {"disabled": sorted(_disabled), "citations": _citations},
            ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:   # 읽기전용/임시 파일시스템 호스팅에서도 계속 동작
        print(f"[state] 저장 실패(무시): {e}", flush=True)


import threading
_retriever_lock = threading.Lock()


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:        # 워밍업 스레드와 첫 요청이 모델을 이중 로딩(=이중 메모리)하지 않도록
            if _retriever is None:
                _retriever = Retriever()
    return _retriever


@app.on_event("startup")
def _warmup():
    """모델을 백그라운드에서 미리 로드한다.

    로딩(~수십 초)이 startup을 막으면 그동안 포트가 안 열려 Fly 프록시가
    'connection refused'를 낸다(특히 auto-stop 후 깨어날 때). 그래서 스레드로
    분리해 포트를 즉시 열고, 모델은 뒤에서 예열한다. 첫 질문은 로딩이 끝날 때까지만 기다린다.
    """
    def run():
        try:
            get_retriever().search("예열", k=1)
            print("[warmup] 모델 로딩 완료", flush=True)
        except Exception as e:
            print(f"[warmup] 실패: {e}", flush=True)
    threading.Thread(target=run, daemon=True).start()


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.post("/api/ask")
def ask(req: AskRequest):
    r = get_retriever()
    evidence = r.search(req.question, k=TOP_K, exclude=_disabled)

    # 답변 거부: 하이브리드 검색 신호(리랭커 또는 dense+BM25)로 근거 유무 판단
    if r.is_refused(evidence):
        return {"refused": True,
                "answer": "지식 팩에서 이 질문의 근거를 찾지 못했습니다. "
                          "다른 질문을 해보시거나 전문가와 상담하세요.",
                "evidence": [], "verification": []}

    gen = generate(req.question, evidence, backend=r.backend)
    verification = verify(gen["sentences"], evidence, backend=r.backend)
    conflict = detect_source_conflict(evidence)

    # 출처별 인용 기여도 집계(문장 인용 1건 = 1카운트) — 수익분배 지표의 원천
    counted = False
    for v in verification:
        for cnum in v["citations"]:
            did = evidence[cnum - 1]["doc_id"]
            _citations[did] = _citations.get(did, 0) + 1
            counted = True
    if counted:
        _save_state()

    return {
        "refused": False,
        "mode": gen["mode"],
        "answer": gen["answer"],
        "verification": verification,
        "conflict": conflict,
        "evidence": [{
            "n": i + 1,
            "chunk_id": c["chunk_id"],
            "title": c["doc_meta"].get("title", c["doc_id"]),
            "publisher": c["doc_meta"].get("publisher", ""),
            "year": c["doc_meta"].get("year", ""),
            "score": round(c["score"], 3),                       # dense 코사인(검색 유사도)
            "relevance": (round(c["rerank_score"], 3)            # 리랭커 관련성(실제 랭킹 신호)
                          if c.get("rerank_score") is not None else None),
            "license": c["doc_meta"].get("license", ""),
            "lic_class": lic_class(c["doc_meta"].get("license")),
        } for i, c in enumerate(evidence)],
    }


class ToggleRequest(BaseModel):
    doc_id: str
    enabled: bool


@app.get("/api/sources")
def sources():
    """출처 목록: 제목·기관·라이선스 등급·On/Off 상태·누적 인용 횟수."""
    r = get_retriever()
    out = []
    for did, d in r.docs.items():
        m = d["meta"]
        cls = lic_class(m.get("license"))
        out.append({"doc_id": did, "title": m.get("title", did),
                    "publisher": m.get("publisher", ""),
                    "license": m.get("license", ""),
                    "lic_class": cls, "lic_label": LABELS[cls],
                    "enabled": did not in _disabled,
                    "cited": _citations.get(did, 0)})
    out.sort(key=lambda x: (-x["cited"], x["title"]))
    return {"sources": out, "disabled": len(_disabled)}


@app.post("/api/sources/toggle")
def toggle_source(req: ToggleRequest):
    """소스 On/Off — 끄면 검색 대상에서 즉시 제외(구독 해지와 동일 효과)."""
    r = get_retriever()
    if req.doc_id not in r.docs:
        raise HTTPException(404, "해당 출처를 찾을 수 없습니다")
    if req.enabled:
        _disabled.discard(req.doc_id)
    else:
        _disabled.add(req.doc_id)
    _save_state()
    return {"doc_id": req.doc_id, "enabled": req.enabled}


@app.get("/api/stats")
def stats():
    """출처별 인용 기여도 — '인용된 만큼 기여했다'는 수익분배 지표의 데모."""
    r = get_retriever()
    total = sum(_citations.values())
    rows = [{"doc_id": d,
             "title": (r.docs[d]["meta"].get("title", d) if d in r.docs else d),
             "cited": c, "share": round(c / total, 3) if total else 0.0}
            for d, c in sorted(_citations.items(), key=lambda x: -x[1])]
    return {"total": total, "rows": rows}


@app.get("/graph")
def graph_page():
    return FileResponse(ROOT / "frontend" / "graph.html")


@app.get("/api/graph")
def graph_data(sim: float = 0.82):
    """지식 팩을 그래프로: 노드=문서/청크, 엣지=소속·의미유사도·출처충돌.

    - member : 문서 → 소속 청크
    - sim    : 같은 문서 청크 간 유사(코사인 ≥ sim)
    - xsim   : 다른 문서 청크 간 유사(= 여러 출처가 같은 사실을 뒷받침 → 신뢰 강화)
    - conflict: 다른 문서 청크가 같은 주제인데 극성이 반대(= 출처 간 상충)
    """
    r = get_retriever()
    chunks, docs = r.chunks, r.docs
    V = np.asarray(r.vectors, dtype=np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Vn = V / norms

    doc_ids = list(docs.keys())
    doc_idx = {d: i for i, d in enumerate(doc_ids)}
    nodes = []
    for d in doc_ids:
        meta = docs[d]["meta"]
        nodes.append({
            "id": f"doc:{d}", "type": "doc", "label": meta.get("title", d),
            "doc": d, "docIdx": doc_idx[d], "publisher": meta.get("publisher", ""),
            "license": meta.get("license", ""), "url": meta.get("url", ""),
            "size": sum(1 for c in chunks if c["doc_id"] == d),
        })

    n = len(chunks)

    # 청크가 많으면(대형 PDF 코퍼스) 청크 그래프는 헤어볼이 되므로 문서 단위로 축약:
    # 문서 벡터 = 소속 청크 임베딩 평균 → 문서 간 kNN 의미 연결.
    if n > 120:
        doc_mat = []
        for d in doc_ids:
            idxs = [i for i, c in enumerate(chunks) if c["doc_id"] == d]
            v = Vn[idxs].mean(axis=0)
            nv = np.linalg.norm(v)
            doc_mat.append(v / (nv if nv else 1.0))
        M = np.stack(doc_mat)
        S2 = M @ M.T
        K = 3
        link = {}
        for i in range(len(doc_ids)):
            order = [int(j) for j in np.argsort(-S2[i]) if int(j) != i][:K]
            for j in order:
                key = (min(i, j), max(i, j))
                link[key] = max(link.get(key, 0.0), float(S2[i, j]))
        edges = [{"s": f"doc:{doc_ids[i]}", "t": f"doc:{doc_ids[j]}",
                  "kind": "xsim", "w": round(w, 3)} for (i, j), w in link.items()]
        stats = {"docs": len(doc_ids), "chunks": n, "xsim": len(edges),
                 "conflict": 0, "backend": r.backend.kind, "mode": "doc-level"}
        return {"nodes": nodes, "edges": edges, "stats": stats, "docNames": doc_ids}

    for i, c in enumerate(chunks):
        nodes.append({
            "id": f"chunk:{c['chunk_id']}", "type": "chunk", "label": c["text"][:38],
            "doc": c["doc_id"], "docIdx": doc_idx[c["doc_id"]], "idx": i,
        })

    edges = [{"s": f"doc:{c['doc_id']}", "t": f"chunk:{c['chunk_id']}", "kind": "member"}
             for c in chunks]
    S = Vn @ Vn.T
    K = 3  # 각 청크의 최근접 이웃 수(kNN). e5는 절대 코사인이 전반적으로 높아
           #  고정 임계보다 kNN이 읽기 좋은 그래프를 만든다.
    link = {}  # (i,j) -> weight  (무방향, 최대값)
    for i in range(n):
        order = [int(j) for j in np.argsort(-S[i]) if int(j) != i]
        picked = 0
        for j in order:
            if chunks[j]["doc_id"] == chunks[i]["doc_id"]:
                continue
            key = (min(i, j), max(i, j))
            link[key] = max(link.get(key, 0.0), float(S[i, j]))
            picked += 1
            if picked >= K:
                break
    for (i, j), w in link.items():
        edges.append({"s": f"chunk:{chunks[i]['chunk_id']}",
                      "t": f"chunk:{chunks[j]['chunk_id']}", "kind": "xsim", "w": round(w, 3)})

    # 출처 간 상충: 의미적으로 같은 주제(코사인 높음)인데 극성 반대 + 내용어 4+ 공유.
    # (일관된 공공 출처 코퍼스에서는 거의 0이 정상 — 상충 기능은 유지하되 엄격하게)
    sents = [[(x, polarity(x), _content_tokens(x)) for x in _split(c["text"])]
             for c in chunks]
    for i in range(n):
        for j in range(i + 1, n):
            if chunks[i]["doc_id"] == chunks[j]["doc_id"] or float(S[i, j]) < 0.88:
                continue
            hit = any(pa and pb and pa != pb and len(ta & tb) >= 4
                      for sa, pa, ta in sents[i] for sb, pb, tb in sents[j])
            if hit:
                edges.append({"s": f"chunk:{chunks[i]['chunk_id']}",
                              "t": f"chunk:{chunks[j]['chunk_id']}", "kind": "conflict"})

    stats = {
        "docs": len(doc_ids), "chunks": n,
        "xsim": sum(1 for e in edges if e["kind"] == "xsim"),
        "conflict": sum(1 for e in edges if e["kind"] == "conflict"),
        "backend": r.backend.kind,
    }
    return {"nodes": nodes, "edges": edges, "stats": stats,
            "docNames": doc_ids}


class FeedbackRequest(BaseModel):
    question: str
    answer: str = ""
    rating: str            # "up" | "down"
    comment: str = ""
    mode: str = ""


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """베타 피드백 수집 — 질문/답변/평가를 JSONL로 append.

    베타의 핵심 산출물이다(어떤 질문이 오고, 어떤 답이 부족한지). 개인정보가
    섞일 수 있으므로 로컬 파일에만 저장하고 외부로 보내지 않는다.
    """
    import datetime
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
           "question": req.question[:500], "answer": req.answer[:2000],
           "rating": req.rating, "comment": req.comment[:500], "mode": req.mode}
    line = _json.dumps(rec, ensure_ascii=False)
    try:
        with (ROOT / "data" / "feedback.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:                      # 읽기전용 호스팅에서도 서비스는 계속
        print(f"[feedback] 파일 저장 실패: {e}", flush=True)
    # 무료 호스팅은 재시작 시 파일이 초기화되므로 로그가 유일한 사본일 수 있다
    print(f"[FEEDBACK] {line}", flush=True)
    return {"ok": True}


@app.get("/api/source/{chunk_id}")
def source(chunk_id: str):
    src = get_retriever().get_source(chunk_id)
    if src is None:
        raise HTTPException(404, "해당 출처를 찾을 수 없습니다")
    return {
        "meta": src["meta"],
        "body": src["body"],
        "highlight": {"start": src["chunk"]["start"], "end": src["chunk"]["end"]},
    }
