import cv2
import numpy as np
from PIL import Image
import io
import joblib
import os

# 모델 로드 (서버 시작 시 1회만)
_model_path = os.path.join(os.path.dirname(__file__), "../models/forgery_detect_model_jpg_v3.pkl")
_model = joblib.load(os.path.abspath(_model_path))


# ============================================================
# [임계값 설정]
# 아래 값들은 테스트 데이터의 정상/위조 분포를 보고 조정해야 함
# ============================================================
# TH_ELA_HIGH        = 15.0   # 이 값보다 크면 편집/합성 의심
# TH_NOISE_HIGH      = 12.0   # 이 값보다 크면 노이즈 불균일 의심
# TH_RED_LOW         = 0.002  # 이 값보다 작으면 직인 없음 의심
# TH_EDGE_LOW        = 0.04   # 이 값보다 작으면 텍스트 밀도 낮음
# TH_BRIGHTNESS_HIGH = 245.0  # 이 값보다 크면 밝기 비정상
# TH_BRIGHTNESS_LOW  = 120.0  # 이 값보다 작으면 밝기 비정상

TH_ELA_HIGH        = 3.0    # 위조 최소 4.59, 정상 최대 0.20 → 3.0이 적절
TH_NOISE_HIGH      = 3.0    # 유지 (이미 잘 걸림)
TH_EDGE_LOW        = 0.08   # 유지
# TH_BRIGHTNESS_HIGH = 220.0  # 유지 (전부 다 220 넘어서 근거로 쓰기 어려움)
# TH_BRIGHTNESS_LOW  = 150.0  # 유지



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


# ============================================================
# [근거 생성] 추출된 특징값을 해석해 위조 판단 이유를 생성
# Isolation Forest는 직접적인 근거를 제공하지 않으므로,
# 규칙 기반으로 어떤 특징이 비정상인지 해석한다.
# ============================================================
def _generate_reasons(features: list) -> list:
    red_ratio, edge_density, avg_brightness, noise_uniformity, ela_score = features
    reasons = []

    if ela_score > TH_ELA_HIGH:
        reasons.append("이미지 편집 또는 합성 흔적이 감지되었습니다.")

    if noise_uniformity > TH_NOISE_HIGH:
        reasons.append("이미지 구역별 노이즈 패턴이 불균일합니다.")

    # if red_ratio < TH_RED_LOW:
    #     reasons.append("인감 또는 직인으로 추정되는 붉은색 영역이 거의 감지되지 않았습니다.")

    if edge_density < TH_EDGE_LOW:
        reasons.append("문서 내 텍스트 및 선 밀도가 정상 범위보다 낮습니다.")

    # if avg_brightness > TH_BRIGHTNESS_HIGH or avg_brightness < TH_BRIGHTNESS_LOW:
    #     reasons.append("이미지 전체 밝기가 정상 스캔 문서 범위를 벗어났습니다.")

    # 위 조건에 하나도 안 걸렸지만 모델은 위조로 판정한 경우
    if not reasons:
        reasons.append("여러 이미지 특징이 복합적으로 정상 문서 패턴에서 벗어났습니다.")

    return reasons


def predict(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    features = _extract_features(img_bytes)
    if features is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    # ── 임시 디버그 로그 ──────────────────────────────────────────
    # print(f"[DEBUG] red_ratio:        {features[0]:.6f}  (기준: < {TH_RED_LOW})")
    print(f"[DEBUG] edge_density:     {features[1]:.6f}  (기준: < {TH_EDGE_LOW})")
    # print(f"[DEBUG] avg_brightness:   {features[2]:.2f}   (기준: < {TH_BRIGHTNESS_LOW} or > {TH_BRIGHTNESS_HIGH})")
    print(f"[DEBUG] noise_uniformity: {features[3]:.6f}  (기준: > {TH_NOISE_HIGH})")
    print(f"[DEBUG] ela_score:        {features[4]:.6f}  (기준: > {TH_ELA_HIGH})")
    # ────────────────────────────────────────────────────────────

    prediction = _model.predict([features])[0]
    score      = float(_model.decision_function([features])[0])
    is_forged  = bool(prediction != 1)
    reasons    = _generate_reasons(features) if is_forged else []

    return {
        "is_forged": is_forged,
        "result":    "위조 의심" if is_forged else "정상",
        "score":     round(score, 3),
        "reasons":   reasons
    }