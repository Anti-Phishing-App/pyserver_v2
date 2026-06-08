FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 (필요 최소)
'''
mecab: 형태소 분석기

mecab-ko-dic: 한국어 품사 정보 사전

libmecab-dev: 파이썬 패키지(mecab-python3)가 리눅스 엔진과 연동되어 빌드될 때 컴파일을 돕는 개발 라이브러리
'''
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-21-jre-headless \
    mecab \
    mecab-ko-dic \
    libmecab-dev \
 && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드
COPY . .

EXPOSE 8001

# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
