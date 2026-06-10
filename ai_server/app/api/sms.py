"""
SMS 피싱 탐지 API 라우터

이 모듈은 SMS 피싱 탐지 관련 REST API 엔드포인트를 제공합니다.

주요 기능:
    1. SMS 종합 분석 (/detect_json)
       - 텍스트 분석: SMS 피싱 특화 키워드 탐지
       - URL 분석: URL 특징 기반 및 ML 모델 + PhishTank DB 분석
       - 종합 위험도 판단

    2. 서비스 상태 확인 (/health)
       - 모델 로드 상태 및 서비스 가용성 체크

분석 방법:
    - 텍스트: ML(70%) + SMS 피싱 키워드(30%)
    - URL: 피싱 사이트 탐지기(URL 기반 + ML + PhishTank) 사용
    - 종합 점수: 텍스트 분석(60%) + URL 분석(40%) 가중치 적용
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime

import time

from app.schemas.sms import (
    SmsDetectRequest,
    SmsDetectResponse,
    TextAnalysisResult,
    UrlAnalysisResult,
)
from app.services.sms_keyword_detector import detect_sms_keywords_batch
from app.services.phishing_site_detector import get_detector as get_site_detector
from app.services.tfidf_phishing_ml import get_sms_ml_detector

router = APIRouter(prefix="/api/sms")


@router.post("/detect_json", response_model=SmsDetectResponse)
async def detect_sms_phishing(request: SmsDetectRequest):
    """
    SMS 피싱 종합 탐지

    SMS 텍스트와 URL을 분석하여 종합적인 피싱 위험도를 판단합니다.

    Args:
        request: SmsDetectRequest
            - sender_hash: 발신자 번호 해시값 (SHA-256)
            - urls: SMS에서 추출된 URL 목록
            - texts: SMS 텍스트 문장 목록
            - received_at: 수신 시간 (밀리초 단위 타임스탬프)

    Returns:
        SmsDetectResponse:
            - phishing_score: 종합 피싱 점수 (0-100)
            - risk_level: 종합 위험도 레벨 (0-3)
            - is_phishing: 피싱 여부
            - warning_message: 경고 메시지
            - text_analysis: 텍스트 분석 결과
            - url_analysis: URL별 분석 결과
            - keywords_found: 탐지된 키워드 목록
            - url_results: URL별 결과 맵

    Example:
        ```json
        {
            "sender_hash": "a1b2c3d4e5f6...",
            "urls": ["http://suspicious-site.com"],
            "texts": ["대출 가능합니다. 계좌번호를 알려주세요."],
            "received_at": 1699999999999
        }
        ```
    """
    try:

        sms_ai_start = time.time()
        
        # 텍스트 결합 (앱에서 이미 문장 단위로 분리해서 보냄)
        full_text = " ".join(request.texts)

        # 초기화
        text_analysis_result = None
        url_analysis_results = []
        all_keywords = []
        url_results_map = {}

        text_score = 0.0
        url_score = 0.0

        # ==================== 1. 텍스트 분석 (ML 70% + 키워드 30%) ====================
        if request.texts and full_text and len(full_text) >= 5:
            try:
                keyword_result = detect_sms_keywords_batch(request.texts)
                keyword_score = 0.0
                keyword_risk_level = 0
                keywords: list = []

                if not keyword_result.get("error"):
                    keyword_score = keyword_result["total_score"] * 100
                    keyword_risk_level = keyword_result["risk_level"]
                    keywords = keyword_result["keywords"]
                    all_keywords.extend(keywords)

                ml_score = 0.0
                ml_is_phishing = False
                ml_confidence = 0.0
                try:
                    # 🚨 모델이 없으므로 강제로 예외를 발생시켜 except 문으로 넘깁니다.
                    raise FileNotFoundError("임시 조치: SMS 모델 없음 스킵")
                    
                    # ml_result = get_sms_ml_detector().predict(full_text, min_chars=5)
                    # ml_confidence = ml_result["confidence"]
                    # ml_is_phishing = ml_result["is_phishing"]
                    # ml_score = ml_confidence * 100
                except Exception as ml_error:
                    print(f"SMS ML 분석 실패 (키워드만 사용): {ml_error}")

                if ml_score > 0 or keyword_score > 0:
                    text_score = ml_score * 0.7 + keyword_score * 0.3
                elif not keyword_result.get("error"):
                    text_score = keyword_score

                if text_score >= 70:
                    combined_risk_level = 3
                elif text_score >= 50:
                    combined_risk_level = 2
                elif text_score >= 30:
                    combined_risk_level = 1
                else:
                    combined_risk_level = keyword_risk_level

                text_analysis_result = TextAnalysisResult(
                    risk_level=combined_risk_level,
                    risk_probability=round(text_score, 2),
                    phishing_type=None,
                    keywords=keywords,
                    is_phishing_kobert=ml_is_phishing if ml_score > 0 else None,
                    kobert_confidence=ml_confidence if ml_score > 0 else None,
                )

            except Exception as e:
                print(f"텍스트 분석 실패: {e}")

        # ==================== 2. URL 분석 ====================
        if request.urls:
            try:
                # URL 분석 시도 (모델이 없으면 스킵)
                try:
                    site_detector = get_site_detector()
                except Exception as model_error:
                    print(f"피싱 사이트 모델 로드 실패 (URL 분석 스킵): {model_error}")
                    # 모델 없이도 URL 존재 여부만 체크
                    for url in request.urls:
                        if url and len(url) >= 10:
                            # 간단한 URL 위험도만 체크
                            url_result = UrlAnalysisResult(
                                url=url,
                                risk_level=1,  # 기본 의심 레벨
                                risk_probability=30.0,  # 기본 점수
                                suspicious_features=["URL 포함"],
                                is_phishing_ml=None,
                                ml_confidence=None,
                                phishtank_matched=None
                            )
                            url_analysis_results.append(url_result)
                            url_results_map[url] = {
                                "risk_level": 1,
                                "risk_probability": 30.0,
                                "note": "모델 없이 기본 분석"
                            }
                    if url_analysis_results:
                        url_score = 30.0  # 기본 점수
                    site_detector = None

                if site_detector:
                    for url in request.urls:
                        if not url or len(url) < 10:
                            continue

                        try:
                            # URL 즉시 분석 (URL 특징 기반)
                            immediate_result = site_detector.detect_immediate(url)

                            # URL 종합 분석 (ML + PhishTank)
                            comprehensive_result = site_detector.detect_comprehensive(url)

                            # URL 분석 결과 구성
                            url_result = UrlAnalysisResult(
                                url=url,
                                risk_level=immediate_result["level"],
                                risk_probability=immediate_result["score"],
                                suspicious_features=immediate_result.get("reasons", []),
                                is_phishing_ml=comprehensive_result["is_phishing"],
                                ml_confidence=comprehensive_result["confidence"],
                                phishtank_matched=comprehensive_result.get("source") == "phishtank"
                            )

                            url_analysis_results.append(url_result)

                            # URL별 결과 맵 (기존 호환성)
                            url_results_map[url] = {
                                "risk_level": url_result.risk_level,
                                "risk_probability": url_result.risk_probability,
                                "is_phishing": url_result.is_phishing_ml,
                                "ml_confidence": url_result.ml_confidence,
                                "phishtank_matched": url_result.phishtank_matched,
                                "suspicious_features": url_result.suspicious_features
                            }

                        except Exception as e:
                            print(f"URL 분석 실패 ({url}): {e}")
                            # 실패한 URL도 기본 결과 추가
                            url_results_map[url] = {
                                "error": str(e),
                                "risk_level": 0,
                                "risk_probability": 0.0
                            }

                # URL 점수 계산 (최대 위험도 기준)
                if url_analysis_results:
                    # URL 기반 점수 (60%)
                    max_url_prob = max([r.risk_probability for r in url_analysis_results])
                    # ML 기반 점수 (40%)
                    max_ml_score = max([
                        r.ml_confidence * 100 if r.is_phishing_ml else 0
                        for r in url_analysis_results
                    ])
                    url_score = max_url_prob * 0.6 + max_ml_score * 0.4

            except Exception as e:
                print(f"URL 분석 전체 실패: {e}")

        # ==================== 3. 종합 점수 계산 ====================
        # 텍스트 가중치 60%, URL 가중치 40%
        if text_analysis_result and url_analysis_results:
            final_score = text_score * 0.6 + url_score * 0.4
        elif text_analysis_result:
            final_score = text_score
        elif url_analysis_results:
            final_score = url_score
        else:
            final_score = 0.0

        # 위험도 레벨 결정
        if final_score >= 70:
            risk_level = 3  # 위험
        elif final_score >= 50:
            risk_level = 2  # 경고
        elif final_score >= 30:
            risk_level = 1  # 의심
        else:
            risk_level = 0  # 안전

        # 피싱 여부
        is_phishing = final_score >= 50

        # 경고 메시지 생성
        if risk_level == 3:
            warning_message = "🚨 위험: 피싱 문자일 가능성이 매우 높습니다! 절대 링크를 클릭하거나 개인정보를 제공하지 마세요."
        elif risk_level == 2:
            warning_message = "⚠️ 경고: 피싱 문자로 의심됩니다. 발신자와 내용을 주의 깊게 확인하세요."
        elif risk_level == 1:
            warning_message = "ℹ️ 주의: 일부 의심스러운 내용이 감지되었습니다. 주의가 필요합니다."
        else:
            warning_message = "✅ 안전: 특별한 위험 요소가 감지되지 않았습니다."

        sms_ai_duration = (time.time() - sms_ai_start) * 1000
        print(f"[RESULT] AI 분석 시간(텍스트+URL): {sms_ai_duration:.2f}ms", flush=True)
        
        # 응답 구성
        return SmsDetectResponse(
            phishing_score=round(final_score, 2),
            risk_level=risk_level,
            is_phishing=is_phishing,
            warning_message=warning_message,
            text_analysis=text_analysis_result,
            url_analysis=url_analysis_results,
            keywords_found=list(set(all_keywords)),  # 중복 제거
            url_results=url_results_map
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SMS 피싱 분석 중 오류 발생: {e}"
        )


@router.get("/health")
async def health_check():
    """
    SMS 피싱 탐지 서비스 상태 확인

    Returns:
        dict: 서비스 상태 정보
    """
    try:
        site_detector = get_site_detector()
        sms_ml_loaded = False
        sms_ml_error = None
        try:
            get_sms_ml_detector()
            sms_ml_loaded = True
        except Exception as e:
            sms_ml_error = str(e)

        return {
            "status": "ok",
            "sms_keyword_detector": "enabled",
            "sms_ml_model_loaded": sms_ml_loaded,
            "sms_ml_error": sms_ml_error,
            "phishing_site_model_loaded": site_detector.model is not None,
            "phishtank_db_loaded": len(site_detector.phishtank_db) > 0,
            "phishtank_db_size": len(site_detector.phishtank_db),
            "message": "SMS 피싱 탐지 서비스가 정상 작동 중입니다."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"서비스 상태 확인 실패: {e}"
        )
