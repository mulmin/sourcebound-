# My Private AI — 출처 기반 육아 지식 챗봇 (캡스톤 1단계)

검증된 지식 팩만을 근거로 답변하고, 문장 단위 출처·유사도를 표기하며,
인용 클릭 시 원문 구절을 하이라이트해 보여주는 RAG 챗봇.

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 지식 팩 인덱스 구축 (data/knowledge_pack/*.md → data/index/)
python -m backend.ingest

# 3. 서버 실행
uvicorn backend.main:app --reload

# 4. 브라우저에서 http://localhost:8000 접속
```

API 키 없이도 동작한다(추출 모드). 실제 LLM 답변 생성을 쓰려면:

```bash
set ANTHROPIC_API_KEY=sk-ant-...   # Windows
pip install anthropic
```

더 좋은 검색 품질을 원하면(권장):

```bash
pip install sentence-transformers   # multilingual-e5-small 자동 사용
python -m backend.ingest            # 인덱스 재구축 필요
```

## 구조

```
backend/
  config.py      전역 설정 (청크 크기, 임계치, 모델명)
  embeddings.py  임베딩 백엔드 (sbert ↔ TF-IDF 자동 폴백)
  ingest.py      지식 팩 구축 파이프라인 (오프라인)
  retriever.py   검색 모듈 (Top-k 코사인 유사도)
  generator.py   생성 모듈 (인용 강제 프롬프팅 / 추출 모드 폴백)
  verifier.py    출처 검증 모듈 (문장-근거 유사도 재계산 → 인용 환각 경고)
  main.py        FastAPI 서버
frontend/index.html  채팅 UI + 출처·유사도 배지 + 원문 뷰어
data/knowledge_pack/ 지식 팩 문서 (현재 개발용 샘플 3건)
eval/                검색 성능 평가 (Recall@k)
```

## 지식 팩 문서 추가 방법

`data/knowledge_pack/`에 아래 형식의 `.md` 파일을 넣고 `python -m backend.ingest` 재실행:

```
title: 문서 제목
publisher: 발행 기관
year: 2025
license: 공공누리 제1유형
---
본문…
```

※ 현재 샘플 3건은 개발용이다. 실제 서비스 전에 질병관리청·보건복지부·WHO 등
공공저작물 원문으로 교체할 것 (수행계획서 3.3 데이터 수집 방안 참고).

## 평가

```bash
python -m eval.run_eval   # Recall@4 출력
```

평가 항목(수행계획서 4장): 검색 Recall@k, 인용 정확도, 환각률(RAG on/off 비교), 답변 거부 적정성.

## 검색 구조 (2단계 반영)

검색은 **하이브리드 회수 → RRF 융합 → 리랭킹**으로 동작한다.

- **dense**(임베딩 코사인): 의역에 강함. sbert 미설치 시 TF-IDF(char n-gram) 폴백.
- **BM25**(`backend/lexical.py`): 수치·약물명·월령 등 표면형 일치에 강함. kiwi 형태소 토큰.
- **RRF**: dense·BM25 순위를 융합해 1차 후보 풀(`CANDIDATE_POOL`) 구성.
- **리랭커**(`backend/reranker.py`): 교차 인코더로 "질문에 답이 되는가" 재채점. 미설치 시 융합점수 폴백.
- **거부 게이트**(`Retriever.is_refused`): 리랭커 점수(있으면) 또는 dense+BM25 동반 신호.
  TF-IDF 경로에서는 char n-gram이 false high를 내므로 **BM25(내용어) 신호를 우선**한다.

## 화면 표시 (프론트엔드)

- **문장 단위 검증 배지**: 각 답변 문장 뒤에 인용 `[n]` + 답변-근거 유사도(%)를 인라인 표기.
  모순 의심은 빨강 `⛔ 모순?`, 근거 약함은 노랑 `⚠`, 미인용은 `근거 없음`으로 구분.
- **출처 목록**: 리랭커 **관련성(%)** 과 임베딩 **검색 유사도(%)** 를 함께 표기.
  관련성은 "질문에 답이 되는가"(실제 랭킹 신호), 유사도는 "주제가 비슷한가"라 서로 다르다
  (예: 유사도 높지만 관련성 0% = 주제만 비슷하고 답은 아님).
- **거부/상충**: 근거 없으면 거부 화면, 출처 간 상충 시 상충 문구와 함께 양쪽 원문 비교 안내.

## 성능 노트

- bge-reranker-v2-m3는 CPU에서 질의당 ~수 초가 걸린다. 빠른 데모가 필요하면
  `MPA_DISABLE_RERANKER=1`로 끄면 하이브리드 융합 점수로 폴백한다(거부는 BM25 신호로 동작).

## 알려진 한계 (개발 노트)

- 샘플 지식 팩이 5건뿐이라 Recall@5 = 1.0은 천장효과다 (k ≥ 문서 수).
  실제 평가는 공공 자료 30~50건 수집 후 수행할 것 (실데이터 과제).
- 추출 모드 답변은 문장을 그대로 이어붙이므로 자연스러움이 떨어진다.
  LLM 모드(ANTHROPIC_API_KEY)에서는 인용 강제 프롬프트(JSON 구조화)로 생성한다.
- 검증은 현재 코사인 유사도 기반. 부정문/모순 탐지(부정어 휴리스틱 + NLI)는 3단계에서 보강한다.
