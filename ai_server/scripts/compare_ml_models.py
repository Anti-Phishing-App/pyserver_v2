#!/usr/bin/env python3
"""로컬에서 학습 모델 추론 스모크 테스트 (서버 배포 전/후 비교용)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.tfidf_phishing_ml import get_phone_ml_detector, get_sms_ml_detector  # noqa: E402

DEFAULT_SAMPLES = {
    "phone": [
        ("normal", "네 알겠습니다 내일 뵙겠습니다 감사합니다"),
        ("phishing", "검찰 수사관입니다 안전계좌로 송금하지 않으면 구속됩니다"),
    ],
    "sms": [
        ("normal", "택배가 발송되었습니다 내일 도착 예정입니다"),
        ("phishing", "긴급 대출 승인 계좌번호와 비밀번호를 링크에서 입력하세요"),
    ],
}


def run(kind: str, texts: list[tuple[str, str]]) -> None:
    getter = get_phone_ml_detector if kind == "phone" else get_sms_ml_detector
    detector = getter()
    print(f"\n=== {kind} model ({detector.model_path.name}) threshold={detector.threshold} ===")
    for label, text in texts:
        result = detector.predict(text)
        print(
            f"[{label}] is_phishing={result['is_phishing']} "
            f"confidence={result['confidence']:.4f} len={result['analyzed_length']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["phone", "sms", "all"], default="all")
    args = parser.parse_args()
    kinds = ["phone", "sms"] if args.kind == "all" else [args.kind]
    for kind in kinds:
        run(kind, DEFAULT_SAMPLES[kind])


if __name__ == "__main__":
    main()
