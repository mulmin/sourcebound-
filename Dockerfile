# Hugging Face Spaces (Docker SDK) 배포용
# - HF Spaces는 컨테이너의 7860 포트를 노출한다.
# - 모델 캐시는 쓰기 가능한 경로여야 하므로 /tmp 아래로 지정한다.
FROM python:3.11-slim

# torch/tokenizers 빌드에 필요한 최소 도구
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 모델·데이터 캐시 위치 (HF Spaces에서 쓰기 가능한 경로)
ENV HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

# 의존성 먼저 설치(레이어 캐시 활용)
COPY requirements.txt .
# torch는 CPU 전용 휠로 설치해 이미지 용량을 줄인다.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
      "numpy>=1.24,<2" fastapi uvicorn "scikit-learn>=1.3" \
      "rank-bm25>=0.2" "kiwipiepy>=0.17" \
      "transformers==4.44.2" "sentence-transformers==3.0.1" \
      "tokenizers<0.20" "huggingface_hub<0.25" \
      openai pymupdf

# 애플리케이션 코드 + 인덱스(원본 PDF는 저장소에 없음 — 인덱스만으로 동작)
COPY . .

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
