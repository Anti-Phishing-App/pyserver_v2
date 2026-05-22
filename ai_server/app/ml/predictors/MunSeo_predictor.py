import cv2
import numpy as np
from PIL import Image
import io
import joblib
import os

# 모델 로드 (서버 시작 시 1회만)
_model_path = os.path.join(os.path.dirname(__file__), "../models/forgery_detect_model_jpg_v3.pkl")
_model = joblib.load(os.path.abspath(_model_path))


def _compute_ela(img_bytes: bytes, quality=90) -> float:
    original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    original_arr     = np.array(original,     dtype=np.float32)
    recompressed_arr = np.array(recompressed, dtype=np.float32)
    return float(np.mean(np.abs(original_arr - recompressed_arr) * 10))


def _extract_features(img_bytes: bytes) -> list:
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # 특징 1: 붉은색 비율
    mask_red = cv2.inRange(img, np.array([0, 0, 100]), np.array([80, 80, 255]))
    red_ratio = np.sum(mask_red > 0) / (img.shape[0] * img.shape[1])

    # 특징 2: 엣지 밀도
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_density = np.sum(cv2.Canny(gray, 100, 200) > 0) / (img.shape[0] * img.shape[1])

    # 특징 3: 평균 밝기
    avg_brightness = np.mean(gray)

    # 특징 4: 노이즈 균일도
    h, w = gray.shape
    regions = [
        gray[0:h//2, 0:w//2], gray[0:h//2, w//2:w],
        gray[h//2:h, 0:w//2], gray[h//2:h, w//2:w],
    ]
    noise_uniformity = np.std([np.std(r) for r in regions])

    # 특징 5: ELA
    ela_score = _compute_ela(img_bytes)

    return [red_ratio, edge_density, avg_brightness, noise_uniformity, ela_score]


def predict(image_path: str) -> dict:
    # 파일 경로에서 바이트로 읽기
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    features = _extract_features(img_bytes)
    if features is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    prediction = _model.predict([features])[0]
    score      = float(_model.decision_function([features])[0])

    return {
        "is_forged": bool(prediction != 1),
        "result":    "위조 의심" if prediction != 1 else "정상",
        "score":     round(score, 3)
    }