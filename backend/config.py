"""전역 설정."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """프로젝트 루트 .env 를 os.environ 에 반영(이미 설정된 값은 덮지 않음).

    ANTHROPIC_API_KEY, DATA_GO_KR_KEY 등을 .env 한 곳에서 관리하기 위함.
    (.env 는 .gitignore 로 커밋 제외)
    """
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
DATA_DIR = ROOT / "data"
KNOWLEDGE_PACK_DIR = DATA_DIR / "knowledge_pack"
INDEX_DIR = DATA_DIR / "index"

CHUNK_SIZE = 400          # 청크 길이(자)
CHUNK_OVERLAP = 80        # 청크 간 겹침(자)
TOP_K = 4                 # 최종 근거 청크 수(리랭킹 후)
# 리랭킹 전 1차 후보 수. 리랭커(교차 인코더)는 CPU에서 쌍당 ~1.6초로 가장 비싼 단계라
# 이 값이 곧 응답 시간이다(20개=32초). 하이브리드(BM25+dense) 순위가 이미 정확해
# 상위 6개만 재채점해도 품질이 유지된다 — 실측으로 확인 후 조정할 것.
CANDIDATE_POOL = int(os.environ.get("MPA_CANDIDATE_POOL", "4"))
RRF_K = 60                # RRF(순위 융합) 상수
SIM_WARN_THRESHOLD = 0.45 # 답변-근거 유사도가 이 미만이면 "근거 약함" 경고

# --- 답변 거부 임계치 ---
# 리랭커가 있으면 리랭커 점수로 판정(가장 calibrated), 없으면 dense+BM25 동반 신호.
SIM_REJECT_THRESHOLD = {"sbert": 0.30, "tfidf": 0.06}  # dense 최고 유사도 하한
BM25_REJECT_THRESHOLD = 0.0   # BM25 최고 점수가 이 이하이면 '어휘적 근거 없음'
# 리랭커 거부: 고정 임계 대신 '후보 분포 기반 상대(적응) 판정'(retriever._reranker_refuses).
# 이유: bge-reranker의 sigmoid 점수는 질문 표현에 따라 절대값이 크게 흔들려(같은 의도가
# 0.18~0.96) 고정 컷은 '맥락상 관련 있는데 점수만 낮은' 질문을 오거부한다. 그래서:
#   top ≥ ANSWER_FLOOR   → 명확히 관련, 응답
#   top < HARD_FLOOR     → 명백 무관(비트코인 등), 즉시 거부
#   그 사이              → 최상위가 나머지 후보에서 '뚜렷이 돌출'하면(구별되는 근거가 있으면)
#                          LLM answerability 게이트로 넘겨 실제 내용으로 판정(여기서 거부 안 함),
#                          분포가 평평하면(구별되는 근거 없음) 거부.
# 최종 answerability는 근거를 읽는 LLM 게이트(main.ask)가 맡는다. '앞은 회수율로 열고 끝은 정밀도로 조인다'.
RERANKER_ANSWER_FLOOR = 0.08   # 이 이상이면 명확히 관련 → 응답
RERANKER_HARD_FLOOR = 0.02     # 이 미만이면 명백 무관 → 즉시 거부
RERANKER_STANDOUT_RATIO = 3.0  # 중간대: 최상위가 나머지 중앙값의 이 배수 이상이면 '돌출'
RERANKER_STANDOUT_GAP = 0.02   # 또는 나머지 중앙값과의 절대 격차가 이 이상이면 '돌출'
RERANKER_REJECT_THRESHOLD = RERANKER_HARD_FLOOR  # 하위호환(옛 상수 참조 대비)

# 임베딩 모델 (sentence-transformers 설치 시 사용, 미설치 시 TF-IDF 폴백)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# 리랭커 모델 (sentence-transformers CrossEncoder 설치 시 사용, 미설치 시 융합점수 폴백)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# LLM 생성 모드 — 키가 있는 제공자를 자동 선택(OpenAI 우선 → Anthropic), 둘 다 없으면 추출 모드.
# 모델명은 .env 에서 OPENAI_MODEL / ANTHROPIC_MODEL 로 덮어쓸 수 있다.
# gpt-5.4-mini: 실제 인용 생성 과제로 비교 선정(1.7s/177토큰으로 핵심 금기사항까지 포착,
# gpt-4o-mini보다 3.8배 빠르고 gpt-5.5보다 7배 빠름). 품질 우선이면 .env 에서 gpt-5.5로 교체.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
LLM_MODEL = ANTHROPIC_MODEL  # 하위 호환
