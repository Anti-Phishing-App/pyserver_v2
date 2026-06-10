"""
SMS 피싱 탐지 API 라우터 (신형 AI Overseer 앙상블 적용 본)

주요 기능:
    1. SMS 종합 분석 (/detect_json)
       - 1차 가드: PhishTank CSV 기반 초고속 블랙리스트 매칭 필터링
       - 2차 가드: 4대 ML(RF, SVM, NB, LR) + 구조 분석 관리 모델(XGBoost) 앙상블 추론
    2. 서비스 상태 확인 (/health)
"""
from fastapi import APIRouter, HTTPException
import time
import traceback

from app.schemas.sms import (
    SmsDetectRequest,
    SmsDetectResponse,
    TextAnalysisResult,
)
from app.utils.text_parser import extract_urls
from app.ml.predictors.smishing_predictor import SmishingOverseerPredictor

router = APIRouter(prefix="/api/sms")

# 서버 부팅 시 AI 엔진 및 PhishTank DB 싱글톤 메모리 상주
sms_ai_engine = SmishingOverseerPredictor(threshold=0.10)


@router.post("/detect_json", response_model=SmsDetectResponse)
async def detect_sms_phishing(request: SmsDetectRequest):
    """
    SMS 피싱 종합 탐지 엔드포인트
    """
    try:
        sms_ai_start = time.time()
        
        # 1. 앱에서 배열 형태로 토스한 원본 문자 본문 병합
        full_text = " ".join(request.texts) if request.texts else ""

        # 2. 본문 문자열을 훼손하지 않고 정규식 패턴으로 URL 리스트만 복사 추출
        detected_urls = extract_urls(full_text)

        # 3. AI 오버시어 앙상블 엔진 추론 수행 (1차 DB 대조 -> 2차 AI 판정)
        ai_result = sms_ai_engine.predict(full_text, detected_urls)
        
        # 4. 결과 매핑 데이터 정제
        final_prob = float(ai_result["phishing_prob"] * 100)  # 0~100 스케일링
        is_phishing = bool(ai_result["is_phishing"])
        source = ai_result["source"]

        # 5. 설정 임계치 및 탐지 소스에 따른 위험도 레벨 및 경고 메시지 결정
        if is_phishing:
            if source == "phishtank":
                warning_message = "🚨 위험: PhishTank 블랙리스트 사기 URL 감지!"
            else:
                warning_message = f"🚨 위험: 스미싱 의심 문자 감지! (위험 확률: {final_prob:.2f}%)"
        else:
            warning_message = "✅ 안전: 위험 요소가 감지되지 않았습니다."

        sms_ai_duration = (time.time() - sms_ai_start) * 1000
        print(f"[RESULT] AI 분석 완료 (소요시간: {sms_ai_duration:.2f}ms | 피싱여부: {is_phishing} | 소스: {source})", flush=True)
        
        # 6. 📱 [앱 통신 규격 호환 보장] SmsDetectResponse 규격에 완벽 바인딩하여 반환
        return SmsDetectResponse(
            phishing_score=round(final_prob, 2),
            is_phishing=is_phishing,
            warning_message=warning_message,
            keywords_found=detected_urls,  # 앱 UI 화면에 링크 주소를 노출할 수 있도록 추출된 URL 매핑
            url_results={}  # 구형 포맷 파싱 에러 방지용 빈 딕셔너리
        )

    except Exception as e:
        print(f"❌ [AI 서버 내부 에러 발생 원인 확인용]: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"AI 판정 레이어 내부 연산 중 오류 발생: {e}"
        )


@router.get("/health")
async def health_check():
    """
    SMS 피싱 탐지 서비스 상태 확인 (신형 엔진 기준 최적화)
    """
    try:
        return {
            "status": "ok",
            "sms_overseer_engine": "enabled",
            "threshold_configured": sms_ai_engine.threshold,
            "phishtank_db_size": len(sms_ai_engine.phishtank_db),
            "phishtank_db_loaded": len(sms_ai_engine.phishtank_db) > 0,
            "message": "신형 앙상블 스미싱 탐지 서비스가 안정적으로 작동 중입니다."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"서비스 상태 확인 실패: {e}"
        )
