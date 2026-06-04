"""
Android antiphishing_app Gson 스키마에 맞춘 /analyze-audio 응답 빌더.

앱(VoiceData.kt)은 immediate/comprehensive의 필수 필드와 단순 타입만 기대합니다.
Clova STT 원본 JSON 전체를 stt_result에 넣으면 Gson 파싱이 실패할 수 있어 제한합니다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.schemas.voice_phishing import ComprehensiveResult, ImmediateResult


def extract_stt_text(stt_result: Dict[str, Any]) -> str:
    """CLOVA 장문 STT 응답에서 분석용 텍스트를 추출한다."""
    if not stt_result:
        return ""

    text = stt_result.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # 일부 응답은 segments / utterances 등에 텍스트가 있음
    for key in ("segments", "utterances", "sentences"):
        chunks = stt_result.get(key)
        if not isinstance(chunks, list):
            continue
        parts: List[str] = []
        for item in chunks:
            if not isinstance(item, dict):
                continue
            for field in ("text", "utterance", "msg"):
                val = item.get(field)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
                    break
        if parts:
            return " ".join(parts).strip()

    return ""


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _app_safe_stt_result(stt_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    앱 SttResult(result/message/token/progress)에 맞는 부분만 포함.
    객체·배열 필드는 제외해 Gson 타입 오류를 방지한다.
    """
    if not stt_result:
        return None

    safe: Dict[str, Any] = {}
    for key in ("result", "message", "token"):
        val = stt_result.get(key)
        if isinstance(val, str):
            safe[key] = val

    progress = stt_result.get("progress")
    if isinstance(progress, bool):
        pass
    elif isinstance(progress, int):
        safe["progress"] = progress
    elif progress is not None:
        try:
            safe["progress"] = int(progress)
        except (TypeError, ValueError):
            pass

    return safe or None


def default_immediate() -> Dict[str, Any]:
    return {
        "level": 0,
        "probability": 0.0,
        "phishing_type": None,
        "keywords": [],
        "method": "word_based",
    }


def default_comprehensive(analyzed_length: int = 0) -> Dict[str, Any]:
    return {
        "is_phishing": False,
        "confidence": 0.0,
        "method": "tfidf_rf",
        "analyzed_length": int(analyzed_length),
    }


def immediate_to_app(immediate: ImmediateResult) -> Dict[str, Any]:
    return {
        "level": int(immediate.level),
        "probability": float(immediate.probability),
        "phishing_type": immediate.phishing_type,
        "keywords": list(immediate.keywords or []),
        "method": immediate.method or "word_based",
    }


def comprehensive_to_app(comprehensive: ComprehensiveResult) -> Dict[str, Any]:
    return {
        "is_phishing": bool(comprehensive.is_phishing),
        "confidence": float(comprehensive.confidence),
        "method": comprehensive.method or "tfidf_rf",
        "analyzed_length": int(comprehensive.analyzed_length),
    }


def build_analyze_audio_response(
    *,
    text: str,
    stt_result: Optional[Dict[str, Any]] = None,
    immediate: Optional[ImmediateResult] = None,
    comprehensive: Optional[ComprehensiveResult] = None,
    warning_message: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    antiphishing_app VoiceAnalysisResponse 파싱용 JSON.
    immediate / comprehensive 는 항상 non-null 객체로 내려준다.
    """
    raw = stt_result or {}
    transcription: Dict[str, Any] = {
        "text": text or "",
        "confidence": _as_optional_float(raw.get("confidence")),
        "speaker": _as_optional_str(raw.get("speaker")),
    }
    safe_stt = _app_safe_stt_result(raw)
    if safe_stt is not None:
        transcription["stt_result"] = safe_stt

    phishing_analysis: Dict[str, Any] = {
        "immediate": immediate_to_app(immediate) if immediate else default_immediate(),
        "comprehensive": (
            comprehensive_to_app(comprehensive)
            if comprehensive
            else default_comprehensive(len(text or ""))
        ),
    }
    if warning_message:
        phishing_analysis["warning_message"] = warning_message
    if error:
        phishing_analysis["error"] = error

    return {
        "transcription": transcription,
        "phishing_analysis": phishing_analysis,
    }
