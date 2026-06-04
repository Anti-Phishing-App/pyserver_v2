"""
피싱 텍스트 분류 엔진 (통화/SMS 공통)

- PHONE: TF-IDF+RF 또는 KoELECTRA 백엔드 선택 가능
- SMS: 기존 TF-IDF+RF 유지
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ai_server/ (Docker WORKDIR /app)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# pyserver/ repo root (model/ lives here on deploy hosts)
REPO_ROOT = BASE_DIR.parent


def _resolve_phone_koelectra_dir() -> Path:
    """KoELECTRA 디렉터리: env > repo model/ > ai_server/model/."""
    if raw := os.getenv("PHONE_KE_MODEL_DIR", "").strip():
        return Path(raw)

    candidates = [
        REPO_ROOT / "model" / "call(KE)" / "call_koelectra_model_v3",
        BASE_DIR / "model" / "call(KE)" / "call_koelectra_model_v3",
    ]
    for path in candidates:
        if (path / "config.json").is_file():
            return path
    return candidates[0]


class TfidfRfPhishingDetector:
    def __init__(
        self,
        model_path: Path,
        vectorizer_path: Path,
        threshold_env: str,
        default_threshold: float = 0.5,
    ):
        if not model_path.is_file():
            raise FileNotFoundError(f"모델 파일 없음: {model_path}")
        if not vectorizer_path.is_file():
            raise FileNotFoundError(f"벡터라이저 파일 없음: {vectorizer_path}")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

        raw_threshold = os.getenv(threshold_env)
        if raw_threshold is None and threshold_env == "PHONE_ML_THRESHOLD":
            raw_threshold = os.getenv("PHISHING_KOBERT_THRESHOLD")
        if raw_threshold is None and threshold_env == "SMS_ML_THRESHOLD":
            raw_threshold = os.getenv("PHISHING_KOBERT_THRESHOLD")
        self.threshold = float(raw_threshold if raw_threshold is not None else default_threshold)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split())

    def _phishing_class_index(self) -> int:
        classes = list(getattr(self.model, "classes_", [0, 1]))
        return classes.index(1) if 1 in classes else 1

    def predict(self, text: str, min_chars: int = 1) -> Dict:
        normalized = self._normalize_text(text)
        if len(normalized) < min_chars:
            return {
                "is_phishing": False,
                "confidence": 0.0,
                "method": "tfidf_rf",
                "analyzed_length": len(normalized),
            }

        features = self.vectorizer.transform([normalized])
        proba = self.model.predict_proba(features)[0]
        confidence = float(proba[self._phishing_class_index()])
        is_phishing = confidence >= self.threshold

        return {
            "is_phishing": is_phishing,
            "confidence": confidence,
            "method": "tfidf_rf",
            "analyzed_length": len(normalized),
        }


class KoElectraPhishingDetector:
    def __init__(
        self,
        model_dir: Path,
        threshold_env: str,
        default_threshold: float = 0.01,
    ):
        if not model_dir.is_dir():
            raise FileNotFoundError(f"KoELECTRA 모델 디렉터리 없음: {model_dir}")

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        raw_threshold = os.getenv(threshold_env)
        if raw_threshold is None and threshold_env == "PHONE_ML_THRESHOLD":
            raw_threshold = os.getenv("PHISHING_KOBERT_THRESHOLD")
        self.threshold = float(raw_threshold if raw_threshold is not None else default_threshold)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split())

    def predict(self, text: str, min_chars: int = 1) -> Dict:
        normalized = self._normalize_text(text)
        if len(normalized) < min_chars:
            return {
                "is_phishing": False,
                "confidence": 0.0,
                "method": "koelectra",
                "analyzed_length": len(normalized),
            }

        enc = self.tokenizer(
            [normalized],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            proba = torch.softmax(logits, dim=-1)[0, 1].item()

        confidence = float(proba)
        is_phishing = confidence >= self.threshold
        return {
            "is_phishing": is_phishing,
            "confidence": confidence,
            "method": "koelectra",
            "analyzed_length": len(normalized),
        }


_phone_detector: Optional[TfidfRfPhishingDetector] = None
_sms_detector: Optional[TfidfRfPhishingDetector] = None
_phone_error: Optional[Exception] = None
_sms_error: Optional[Exception] = None


def get_phone_ml_detector() -> TfidfRfPhishingDetector:
    global _phone_detector, _phone_error
    if _phone_detector is not None:
        return _phone_detector
    if _phone_error is not None:
        raise _phone_error
    try:
        backend = os.getenv("PHONE_ML_BACKEND", "koelectra").strip().lower()
        if backend == "koelectra":
            model_dir = _resolve_phone_koelectra_dir()
            _phone_detector = KoElectraPhishingDetector(
                model_dir=model_dir,
                threshold_env="PHONE_ML_THRESHOLD",
                default_threshold=0.01,
            )
        elif backend in {"tfidf_rf", "rf"}:
            model_dir = Path(os.getenv("PHONE_MODEL_DIR", BASE_DIR / "data/models/phone"))
            _phone_detector = TfidfRfPhishingDetector(
                model_path=Path(os.getenv("PHONE_MODEL_PATH", model_dir / "phone_phishing_model.pkl")),
                vectorizer_path=Path(
                    os.getenv("PHONE_VECTORIZER_PATH", model_dir / "phone_tfidf_vectorizer.pkl")
                ),
                threshold_env="PHONE_ML_THRESHOLD",
                default_threshold=0.5,
            )
        else:
            raise ValueError(
                f"PHONE_ML_BACKEND must be 'koelectra' or 'tfidf_rf', got: {backend!r}"
            )
        return _phone_detector
    except Exception as exc:
        _phone_error = exc
        raise


def get_sms_ml_detector() -> TfidfRfPhishingDetector:
    global _sms_detector, _sms_error
    if _sms_detector is not None:
        return _sms_detector
    if _sms_error is not None:
        raise _sms_error
    try:
        model_dir = Path(os.getenv("SMS_MODEL_DIR", BASE_DIR / "data/models/sms"))
        _sms_detector = TfidfRfPhishingDetector(
            model_path=Path(os.getenv("SMS_MODEL_PATH", model_dir / "sms_phishing_model.pkl")),
            vectorizer_path=Path(
                os.getenv("SMS_VECTORIZER_PATH", model_dir / "sms_tfidf_vectorizer.pkl")
            ),
            threshold_env="SMS_ML_THRESHOLD",
            default_threshold=0.5,
        )
        return _sms_detector
    except Exception as exc:
        _sms_error = exc
        raise
