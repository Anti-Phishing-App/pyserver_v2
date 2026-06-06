import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import shutil
import os

from app.services.document_service import analyze_document
from app.api import voice_phishing, phishing_site, sms, transcribe, transcribe_stream

# 로깅 설정 (AI 서버용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_server")

app = FastAPI(title="AI Analysis Worker Server", version="1.0.0")

# 분석용 임시 디렉토리
TEMP_DIR   = "temp_analysis"
UPLOAD_DIR = "uploaded_images"   # ── 추가: 이미지 서빙용 디렉토리
os.makedirs(TEMP_DIR,   exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── 추가: 업로드된 이미지 static 서빙 ────────────────────────────────
app.mount("/uploaded_images", StaticFiles(directory=UPLOAD_DIR), name="uploaded_images")


@app.post("/process-request")
def analyze_doc_task(
    file: UploadFile = File(...),
    force: bool = False          # ── 문서 판별 건너뛰고 강제 검사
):
    """
    API 서버로부터 파일을 전달받아 AI 모델로 분석을 수행합니다.
    force=True 이면 문서 판별을 건너뛰고 바로 위조 분석을 수행합니다.
    """
    file_path   = os.path.join(TEMP_DIR,   file.filename)
    upload_path = os.path.join(UPLOAD_DIR, file.filename)  # ── 추가
    try:
        # 1. 파일 임시 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. 프론트에서 이미지 불러올 수 있도록 uploaded_images에 복사
        shutil.copy2(file_path, upload_path)

        # 3. 🔥 AI 모델 실행 (force 파라미터 전달)
        logger.info(f"Starting analysis for: {file.filename} (force={force})")
        result = analyze_document(file_path, force=force)

        return result
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # temp_analysis 파일만 삭제 (uploaded_images는 프론트에서 불러와야 하므로 유지)
        if os.path.exists(file_path):
            os.remove(file_path)


# AI 연산 라우터는 AI 서버에서 직접 제공
app.include_router(transcribe.router, tags=["Transcribe"])
app.include_router(transcribe_stream.router)
app.include_router(voice_phishing.router, tags=["Voice Phishing Detection"])
app.include_router(phishing_site.router, tags=["Phishing Site Detection"])
app.include_router(sms.router, tags=["SMS Phishing Detection"])


@app.get("/healthz")
def healthz():
    """AI 서버 생존 확인용"""
    return {"status": "ok", "service": "ai_server"}