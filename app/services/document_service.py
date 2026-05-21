"""문서 분석 서비스"""
from pathlib import Path
from fastapi import HTTPException

from app.ml.predictors.ocr_predictor import run_ocr, OCRError
from app.ml.predictors.keyword_predictor import detect_keywords
from app.ml.predictors.layout_predictor import analyze_document_font
from app.ml.predictors.stamp_predictor import run_stamp_detection
from app.ml.predictors.MunSeo_predictor import predict as run_forgery_detection


def analyze_document(image_path: Path) -> dict:
    """
    문서 이미지 전체 분석

    Args:
        image_path: 분석할 이미지 경로

    Returns:
        분석 결과 딕셔너리
    """
    try:
        # ── 기존 분석 방식 (주석처리) ────────────────────────────────
        # stamp_result   = run_stamp_detection(str(image_path))
        # ocr_result     = run_ocr(str(image_path))
        # keyword_result = detect_keywords(ocr_result)
        # layout_result  = analyze_document_font(ocr_result)

        # stamp_score   = stamp_result.get("score", 0) or 0.0
        # keyword_score = keyword_result.get("total_score", 0) or 0.0
        # layout_score  = layout_result.get("score", 0) or 0.0

        # 가중치 부여: 직인 30%, 키워드 50%, 레이아웃 20%
        # final_risk = round((stamp_score * 0.3) + (keyword_score * 0.5) + (layout_score * 0.2), 2)

        # return {
        #     "stamp": stamp_result,
        #     "keyword": keyword_result,
        #     "layout": layout_result,
        #     "final_risk": final_risk
        # }

        # ── 새로운 분석 방식 (머신러닝 기반 위조 탐지) ───────────────
        forgery_result = run_forgery_detection(str(image_path))

        return {
            "forgery": forgery_result  # is_forged, result, score
        }

    except OCRError as e:
        raise HTTPException(status_code=500, detail=f"OCR 처리 실패: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"알 수 없는 서버 오류: {e}")