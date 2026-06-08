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

from app.schemas.sms import (
    SmsDetectRequest,
    SmsDetectResponse,
    TextAnalysisResult,
)
from app.utils.text_parser import extract_urls  # 선배 유틸에서 URL 추출 기능 계승
from app.ml.predictors.smishing_predictor import SmishingOverseerPredictor

router = APIRouter(prefix="/api/sms")

# 🎯 [인프라 최적화] 서버 기동 시 코치님의 AI 엔진 및 PhishTank DB를 단 한 번만 메모리에 로드 (싱글톤)
sms_ai_engine = SmishingOverseerPredictor(threshold=0.10)


@router.post("/detect_json", response_model=SmsDetectResponse)
async def detect_sms_phishing(request: SmsDetectRequest):
    """
    SMS 피싱 종합 탐지 (하이브리드 AI 주심 모델)
    """
    try:
        sms_ai_start = time.time()
        
        # 1. 앱에서 분리해서 보낸 문장 배열을 하나의 문자열 원본으로 결합 (맥락 보존)
        full_text = " ".join(request.texts)

        # 2. 본문 문자열을 훼손하지 않고 정규식 패턴으로 URL 리스트만 복사 추출
        detected_urls = extract_urls(full_text)

        # 3. 💥 신형 엔진 가동 (1차 DB 대조 ➡️ 2차 AI 앙상블 자동 수행)
        ai_result = sms_ai_engine.predict(full_text, detected_urls)
        
        # 4. 결과 매핑 데이터 정제
        final_prob = ai_result["phishing_prob"] * 100  # 0~100 스케일 변환
        is_phishing = ai_result["is_phishing"]
        source = ai_result["source"]

        # 5. 설정 임계치 및 탐지 소스에 따른 위험도 레벨 및 경고 메시지 결정
        if is_phishing:
            risk_level = 3  # 위험
            if source == "phishtank":
                warning_message = "🚨 위험: PhishTank 블랙리스트에 등록된 사기 URL이 포함되어 있습니다! 절대 클릭하지 마세요."
            else:
                warning_message = f"🚨 위험: AI 탐지 스미싱 위험 문자입니다! (위험 확률: {final_prob:.2f}%)"
        else:
            risk_level = 0  # 안전
            warning_message = "✅ 안전: 특별한 위험 요소가 감지되지 않았습니다."

        sms_ai_duration = (time.time() - sms_ai_start) * 1000
        print(f"[RESULT] AI 분석 완료 (소요시간: {sms_ai_duration:.2f}ms, 소스: {source})", flush=True)
        
        # 6. 📱 [앱 통신 규격 호환 보장] SmsDetectResponse 규격에 완벽 바인딩하여 반환
        return SmsDetectResponse(
            phishing_score=round(final_prob, 2),
            risk_level=risk_level,
            is_phishing=is_phishing,
            warning_message=warning_message,
            
            # 하위 호환성을 위해 내부 분석 껍데기 포맷 매핑
            text_analysis=TextAnalysisResult(
                risk_level=risk_level,
                risk_probability=round(final_prob, 2),
                phishing_type=None,
                keywords=[],
                is_phishing_kobert=is_phishing if source == "overseer_ai" else None,
                kobert_confidence=ai_result["phishing_prob"] if source == "overseer_ai" else None
            ),
            url_analysis=[],  # 무거운 실시간 HTML 크롤링을 전면 제거했으므로 빈 값 처리
            keywords_found=detected_urls,  # 앱 UI 화면에 링크 주소를 노출할 수 있도록 추출된 URL 매핑
            url_results={}  # 구형 포맷 파싱 에러 방지용 빈 딕셔너리
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"신형 AI 서빙 레이어 추론 중 오류 발생: {e}"
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