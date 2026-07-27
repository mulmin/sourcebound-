"""생성 모듈: 인용 강제 프롬프팅으로 LLM 호출.

ANTHROPIC_API_KEY가 설정되어 있으면 실제 LLM을 호출하고,
없으면 '추출 모드'(근거 청크의 핵심 문장을 인용과 함께 그대로 제시)로
동작하여 API 키 없이도 전체 데모가 가능하다.

출력은 항상 '구조화된 문장 목록'으로 반환한다:
    {"mode", "sentences": [{"text", "citations": [n, ...]}], "answer": str}
문장과 인용을 데이터로 보존하므로, 검증 모듈이 정규식으로 문장을
추측 분할할 필요가 없다(인용-문장 매핑이 생성 의도 그대로 유지된다).
"""
from __future__ import annotations
import json
import os
import re

from backend.config import LLM_MODEL

ADVISORY = "정확한 판단이 필요하면 소아청소년과 전문의와 상담하세요."

# LLM에 JSON 구조화 출력을 강제한다(문장 단위 인용 보존을 위해).
CITATION_PROMPT = """당신은 육아 지식 도우미입니다. 부모에게 이야기하듯 간결하고 자연스러운 상담체로 답하세요.
아래 규칙을 반드시 지키세요.

1. 오직 제공된 [근거]만 사용해 답하세요. 외부 지식·추측 금지.
2. 답변을 문장 단위로 쪼개고, 각 문장이 근거한 번호를 citations 배열에 담으세요.
3. 근거에 없는 내용은 답하지 말고, 답할 근거가 전혀 없으면 sentences를 빈 배열로 두세요.
   특히 질문이 영유아·육아와 무관하거나(예: 성인 질환, 반려동물, 투자, 날씨, 프로그래밍 등),
   제공된 근거가 질문에 직접 답하지 않으면 반드시 sentences를 빈 배열([])로 두세요.
   근거가 '주제만 비슷하고' 실제 답이 아니면 억지로 답하지 마세요.
4. 〔메타 문장 금지〕 "제공된 근거에는 ~에 대한 정보가 없습니다", "자료에 ~라는 내용이 없습니다"처럼
   자료 자체를 언급하거나 '근거가 없다'고 서술하는 문장은 절대 쓰지 마세요.
   답할 내용이 있으면 그 내용만 자연스럽게 쓰고, 없으면 sentences를 빈 배열([])로 두세요.
5. 〔관련 문장만〕 질문에 직접 답하는 문장만 담으세요. 근거에 들어 있어도 질문과 주제가 다른 문장
   (예: 발열 질문에 트림·수유 설명)은 넣지 마세요.
6. 의학적 판단이 필요한 사안은 마지막에 전문가 상담 권고 문장을 citations 없이 덧붙이세요.
7. 〔원문 그대로만 인용〕 표시가 붙은 근거는 저작권상 변경이 금지된 자료입니다.
   그 근거의 내용은 요약·의역하지 말고 원문 문장을 그대로 옮겨 인용하세요.

반드시 아래 JSON 형식으로만 출력하세요(설명·코드펜스 금지):
{{"sentences": [{{"text": "문장", "citations": [1]}}, {{"text": "...", "citations": [2, 3]}}]}}

[근거]
{evidence}

[질문]
{question}
"""

# 스트리밍용: JSON 대신 '평문 + [n] 인용'을 흘려보내 토큰을 실시간 표시한다.
# 끝나면 _structure_from_text 로 문장·인용을 복원한다(검증/배지용).
STREAM_SENTINEL = "__NO_ANSWER__"
CITATION_PROMPT_STREAM = """당신은 육아 지식 도우미입니다. 부모에게 이야기하듯 간결하고 자연스러운 상담체로 답하세요.
아래 규칙을 반드시 지키세요.

1. 오직 제공된 [근거]만 사용해 답하세요. 외부 지식·추측 금지.
2. 각 문장 끝에 근거 번호를 대괄호로 답니다. 예: "…하는 것이 좋습니다.[1]" 여러 근거면 [1][2].
3. 근거에 없거나, 질문이 영유아·육아와 무관하거나(성인 질환·반려동물·투자 등), 근거가 질문에
   직접 답하지 않으면 아무 문장도 쓰지 말고 정확히 '__NO_ANSWER__' 한 줄만 출력하세요.
4. "자료에 없다"처럼 자료 자체를 언급하는 메타 문장 금지. 질문에 직접 답하는 문장만.
5. 질문과 주제가 다른 근거 문장은 넣지 마세요.
6. 의학적 판단이 필요하면 마지막에 전문가 상담 권고를 번호 없이 덧붙이세요.
7. 〔원문 그대로만 인용〕 표시가 붙은 근거는 원문 문장을 그대로 옮겨 인용하세요.

[근거]
{evidence}

[질문]
{question}

답변:"""

CITE_RE = re.compile(r"\[(\d+)\]")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.다요])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# 청크 경계에서 잘린 '토막 문장' 판별용
_TAIL_OK = re.compile(r"(?:다|요|음|임)[.!?]?$|[.!?]$")          # 제대로 끝나는가
_HEAD_BAD = re.compile(r"^(?:[,)\]·•]|고,|며,|서,|은|는|을|를|이|가)\s")  # 중간에서 시작하나


# 문장 첫머리에 올 수 있는 접속부사(연결어미로 끝나 보이지만 정상)
_HEAD_OK_WORDS = {"그리고", "그러나", "하지만", "또한", "따라서", "그래서",
                  "만약", "다만", "특히", "예를", "만일", "그러므로"}
_kiwi = None


def _starts_midclause(s: str) -> bool:
    """첫 어절이 '연결어미(EC)'로 끝나면 앞 문장에서 이어진 토막으로 본다.
    (예: '있고 심하면…', '되어 …') — 형태소 분석으로 판정."""
    global _kiwi
    head = s.split()[0] if s.split() else ""
    if not head or head in _HEAD_OK_WORDS:
        return False
    try:
        if _kiwi is None:
            from kiwipiepy import Kiwi
            _kiwi = Kiwi()
        toks = _kiwi.tokenize(head)
    except Exception:
        return False
    if not toks:
        return False
    last = toks[-1].tag
    # EC(연결어미)로 끝나거나, 짧은 어절이 관형사형 어미(ETM)로 끝나면 토막 의심
    return last == "EC" or (last == "ETM" and len(head) <= 3)


def _is_wellformed(s: str) -> bool:
    """완결된 문장인지. PDF/청크 경계로 잘린 토막을 걸러 답변 가독성을 지킨다."""
    if len(s) < 12:
        return False
    if not _TAIL_OK.search(s):
        return False
    if _HEAD_BAD.match(s):
        return False
    if _starts_midclause(s):
        return False
    return True


def _clean_sentences(text: str, wellformed_only: bool = False) -> list[str]:
    """청크를 문장으로 나누되, 소제목 같은 '짧고 종결어미 없는 줄'은 제외한다.
    wellformed_only=True 면 잘린 토막 문장까지 걸러낸다(추출 답변용)."""
    out = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) < 20 and not re.search(r"[.다요]$", para):
            continue  # 헤더로 간주하고 건너뜀
        out.extend(_split_sentences(para))
    if wellformed_only:
        out = [s for s in out if _is_wellformed(s)]
    return out


def _render(sentences: list[dict]) -> str:
    """구조화 문장 → '문장[1][2] 문장[3]' 형태의 표시용 문자열."""
    out = []
    for s in sentences:
        cites = "".join(f"[{c}]" for c in s.get("citations", []))
        out.append(f"{s['text']}{cites}")
    return " ".join(out)


def _structure_from_text(text: str, n_evidence: int) -> list[dict]:
    """LLM이 JSON 대신 평문을 준 경우의 폴백: 평문을 문장+인용으로 복원."""
    sentences = []
    for raw in _split_sentences(text):
        cites = [int(m) for m in CITE_RE.findall(raw)
                 if 1 <= int(m) <= n_evidence]
        clean = CITE_RE.sub("", raw).strip()
        if clean:
            sentences.append({"text": clean, "citations": sorted(set(cites))})
    return sentences


def _call_llm(prompt: str) -> str | None:
    """키가 있는 제공자로 생성 호출. OpenAI 우선 → Anthropic. 둘 다 없거나 실패하면 None
    (호출 측이 추출 모드로 폴백 — '키 없이도 끝까지 도는' 설계 유지)."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            from backend.config import OPENAI_MODEL
            client = OpenAI()
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},  # 구조화 출력 강제
            )
            return r.choices[0].message.content
        except Exception as e:
            print(f"OpenAI 호출 실패: {e}")

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            from backend.config import ANTHROPIC_MODEL
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            print(f"Anthropic 호출 실패: {e}")

    return None


def _parse_llm_json(text: str, n_evidence: int) -> list[dict] | None:
    """LLM 출력에서 JSON 문장 목록을 추출. 실패 시 None."""
    snippet = text.strip()
    if snippet.startswith("```"):
        snippet = snippet.strip("`")
        snippet = snippet.split("\n", 1)[-1] if "\n" in snippet else snippet
    start, end = snippet.find("{"), snippet.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(snippet[start:end + 1])
        raw = obj.get("sentences", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    sentences = []
    for s in raw:
        if not isinstance(s, dict) or not s.get("text"):
            continue
        cites = [int(c) for c in s.get("citations", [])
                 if str(c).isdigit() and 1 <= int(c) <= n_evidence]
        sentences.append({"text": str(s["text"]).strip(),
                          "citations": sorted(set(cites))})
    return sentences


def _extractive(question: str, evidence: list[dict], backend) -> list[dict]:
    """추출 모드(폴백): 근거에서 질의와 가장 가까운 문장을 인용과 함께 구성.

    자연스러움·일관성을 위해 **상위 근거(리랭커 최상위)와 같은 문서의 청크만** 사용한다.
    서로 다른 주제의 문서를 섞으면 상반된 지식이 이어붙는 문제가 생기기 때문이다.
    (진짜 자연스러운 서술은 LLM 모드가 담당; 추출은 키 없을 때의 폴백이다.)"""
    from backend.embeddings import EmbeddingBackend, cosine
    top_doc = evidence[0]["doc_id"]
    pool = [(i, c) for i, c in enumerate(evidence) if c["doc_id"] == top_doc]

    sents, owners, seen = [], [], set()
    for i, c in pool:
        for s in _clean_sentences(c["text"], wellformed_only=True):
            key = "".join(s.split())
            if len(s) > 15 and key not in seen:   # 청크 겹침 중복 제거
                seen.add(key)
                sents.append(s)
                owners.append(i)
    if not sents:   # 완결 문장이 없으면 기준을 낮춰 재시도(빈 답변 방지)
        for i, c in pool:
            for s in _clean_sentences(c["text"]):
                key = "".join(s.split())
                if len(s) > 15 and key not in seen:
                    seen.add(key); sents.append(s); owners.append(i)
    if not sents:
        return []
    if backend is None:
        backend = EmbeddingBackend(corpus=sents + [question])
    sims = cosine(backend.encode([question], kind="query"),
                  backend.encode(sents, kind="passage"))[0]
    # 질의와 가장 관련된 상위 문장을 고르되, 원문 등장 순서로 배열해 흐름을 유지
    top = sorted(range(len(sents)), key=lambda i: -sims[i])[:3]
    sentences = [{"text": sents[i], "citations": [owners[i] + 1]}
                 for i in sorted(top)]
    return sentences


def _numbered_evidence(evidence: list[dict]) -> str:
    """근거를 '[n] (제목)〔라이선스〕 본문' 형태의 번호 목록 문자열로."""
    from backend.licensing import lic_class, allow_rewrite
    return "\n".join(
        f"[{i+1}] ({c['doc_meta'].get('title', c['doc_id'])})"
        + ("" if allow_rewrite(lic_class(c['doc_meta'].get('license')))
           else "〔원문 그대로만 인용〕")
        + f" {c['text']}"
        for i, c in enumerate(evidence)
    )


def generate(question: str, evidence: list[dict], backend=None) -> dict:
    """반환: {"mode", "sentences": [{"text","citations"}], "answer": str}

    backend: 공유 EmbeddingBackend(추출 모드에서 재사용). None이면 새로 생성.
    """
    numbered = _numbered_evidence(evidence)

    sentences: list[dict] = []
    mode = "extractive"
    prompt = CITATION_PROMPT.format(evidence=numbered, question=question)
    text = _call_llm(prompt)
    if text is not None:
        sentences = _parse_llm_json(text, len(evidence))
        if sentences is None:      # JSON 파싱 실패 → 평문 복원 폴백
            sentences = _structure_from_text(text, len(evidence))
        mode = "llm"

    if mode != "llm":
        sentences = _extractive(question, evidence, backend)

    if not sentences:
        return {"mode": mode, "sentences": [],
                "answer": "제공된 자료에서 근거를 찾지 못했습니다."}

    # 의학적 상담 권고를 항상 마지막에 보강(중복 방지)
    if not any("상담" in s["text"] for s in sentences):
        sentences.append({"text": ADVISORY, "citations": []})

    return {"mode": mode, "sentences": sentences, "answer": _render(sentences)}


def generate_stream(question: str, evidence: list[dict], backend=None):
    """스트리밍 생성기. 순서대로 yield한다:
        {"delta": str}      — 답변 텍스트 조각(실시간 표시용)
        {"refused": True}   — 근거로 답할 수 없음(빈 답변/무관 주제)
        {"final": {...}}    — 스트리밍 종료 후 구조화 결과(검증·배지용)

    OpenAI 스트리밍으로 '평문 + [n]'을 흘리고, 끝나면 문장·인용을 복원한다.
    거부 신호(__NO_ANSWER__)가 화면에 노출되지 않도록 첫 조각은 서버에서 버퍼링한다.
    키가 없거나 실패하면 비스트리밍 generate()로 폴백해 한 번에 전달한다."""
    numbered = _numbered_evidence(evidence)

    def _fallback():
        g = generate(question, evidence, backend=backend)
        if not any(s.get("citations") for s in g["sentences"]):
            yield {"refused": True}
            return
        yield {"delta": g["answer"]}
        yield {"final": g}

    if not os.environ.get("OPENAI_API_KEY"):
        yield from _fallback()
        return

    prompt = CITATION_PROMPT_STREAM.format(evidence=numbered, question=question)
    full, buf, flushing = "", "", False
    try:
        from openai import OpenAI
        from backend.config import OPENAI_MODEL
        stream = OpenAI().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if not delta:
                continue
            full += delta
            if flushing:
                yield {"delta": delta}
                continue
            buf += delta
            if len(buf) >= len(STREAM_SENTINEL):     # 거부 신호인지 판별 가능한 만큼 모임
                if buf.lstrip().startswith(STREAM_SENTINEL):
                    yield {"refused": True}
                    return
                flushing = True
                yield {"delta": buf}
                buf = ""
        if not flushing:                             # 아주 짧은 응답: 아직 안 비운 buf 처리
            if buf.strip() and not buf.lstrip().startswith(STREAM_SENTINEL):
                yield {"delta": buf}
            else:
                yield {"refused": True}
                return
    except Exception as e:
        print(f"OpenAI 스트리밍 실패(폴백): {e}", flush=True)
        yield from _fallback()
        return

    sentences = _structure_from_text(full, len(evidence))
    if not any(s.get("citations") for s in sentences):
        yield {"refused": True}
        return
    if not any("상담" in s["text"] for s in sentences):
        sentences.append({"text": ADVISORY, "citations": []})
    yield {"final": {"mode": "llm", "sentences": sentences,
                     "answer": _render(sentences)}}
