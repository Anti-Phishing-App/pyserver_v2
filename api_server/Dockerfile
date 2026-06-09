FROM python:3.11-slim

WORKDIR /app

# 1. 시스템 패키지 설치
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

# 2. 파이썬이 찾을 수 있도록 mecab 설정 파일 생성 및 사전 경로 지정
RUN mkdir -p /usr/local/etc \
 && echo "dicdir = /usr/local/lib/mecab/dic/mecab-ko-dic" > /usr/local/etc/mecabrc

# 3. 한국어 형태소 사전 다운로드 및 수동 컴파일 (★ 설치 경로 강제 지정 추가!)
RUN cd /tmp \
 && wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz \
 && tar -zxvf mecab-ko-dic-2.1.1-20180720.tar.gz \
 && cd mecab-ko-dic-2.1.1-20180720 \
 && ./autogen.sh \
 && ./configure --with-dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic \
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
