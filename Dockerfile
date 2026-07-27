# Fly.io / 컨테이너 배포용 — sourcebound
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 모델 캐시를 이미지에 구워넣는다(빌드 시 다운로드 → 재기동 때 재다운로드 없음).
# auto-stop으로 서버가 잠들었다 깨어나도 모델이 이미 이미지에 있어 바로 로딩된다.
ENV HF_HOME=/opt/hf \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TOKENIZERS_PARALLELISM=false

# 1) 의존성 (CPU 전용 torch로 이미지 경량화)
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
      "numpy>=1.24,<2" fastapi uvicorn "scikit-learn>=1.3" \
      "rank-bm25>=0.2" "kiwipiepy>=0.17" \
      "transformers==4.44.2" "sentence-transformers==3.0.1" \
      "tokenizers<0.20" "huggingface_hub<0.25" \
      openai pymupdf

# 2) 임베딩·리랭커 모델을 빌드 단계에서 미리 받아 이미지에 포함(재기동 시 재다운로드 없음)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('intfloat/multilingual-e5-small'); CrossEncoder('BAAI/bge-reranker-v2-m3'); print('models cached')"

# 3) 애플리케이션 코드 + 인덱스 (원본 PDF는 .dockerignore로 제외)
COPY . .

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
