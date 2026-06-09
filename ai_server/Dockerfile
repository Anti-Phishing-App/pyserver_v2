FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 및 컴파일 도구 설치
# mecab-ko-dic을 직접 빌드하기 위해 wget, automake, libtool 등을 추가했습니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-21-jre-headless \
    libmecab-dev \
    mecab \
    wget \
    automake \
    libtool \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 한국어 형태소 사전(mecab-ko-dic) 다운로드 및 수동 컴파일 설치
RUN cd /tmp \
 && wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz \
 && tar -zxvf mecab-ko-dic-2.1.1-20180720.tar.gz \
 && cd mecab-ko-dic-2.1.1-20180720 \
 && ./autogen.sh \
 && ./configure \
 && make \
 && make install \
 && ldconfig \
 && rm -rf /tmp/*

# 파이썬 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드
COPY . .

EXPOSE 8001

# FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
