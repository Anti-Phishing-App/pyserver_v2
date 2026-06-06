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
# ============================================================
# TH_ELA_HIGH        = 15.0
# TH_NOISE_HIGH      = 12.0
# TH_RED_LOW         = 0.002
# TH_EDGE_LOW        = 0.04
# TH_BRIGHTNESS_HIGH = 245.0
# TH_BRIGHTNESS_LOW  = 120.0

TH_ELA_HIGH   = 3.0
TH_NOISE_HIGH = 3.0
TH_EDGE_LOW   = 0.08
# TH_BRIGHTNESS_HIGH = 220.0
# TH_BRIGHTNESS_LOW  = 150.0

# ── 문서 판별 임계값 ───────────────────────────────────────────
TH_DOC_BRIGHTNESS_MIN = 180.0
TH_DOC_EDGE_MAX       = 0.35
TH_DOC_WHITE_MIN      = 0.3


# ============================================================
# [문서 영역 크롭]
# 스크린샷 등 불필요한 UI 영역을 제거하고 문서 부분만 추출
# 사각형 윤곽선 중 가장 큰 영역을 문서로 판단
# ============================================================
def _crop_document(img_bytes: bytes) -> bytes:
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return img_bytes

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 가우시안 블러 → 엣지 감지
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 2. 윤곽선 탐지
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bytes

    # 3. 가장 큰 사각형 윤곽선 찾기
    best_contour = None
    best_area    = 0
    img_area     = img.shape[0] * img.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)

        # 너무 작거나(10% 미만) 이미지 전체(98% 초과)인 건 제외
        if area < img_area * 0.1 or area > img_area * 0.98:
            continue

        # 사각형에 가까운 윤곽선만 선택 (꼭짓점 4개)
        peri  = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4 and area > best_area:
            best_area    = area
            best_contour = approx

    if best_contour is None:
        # 사각형 윤곽선 못 찾으면 원본 그대로 반환
        return img_bytes

    # 4. 찾은 사각형 영역만 크롭
    x, y, w, h = cv2.boundingRect(best_contour)
    cropped = img[y:y+h, x:x+w]

    # 5. bytes로 변환해서 반환
    _, encoded = cv2.imencode('.jpg', cropped)
    return encoded.tobytes()


# ============================================================
# [문서 판별]
# 규칙 기반으로 문서의 특징(밝은 배경, 낮은 엣지 밀도)을 확인
# ============================================================
def _is_document(img_bytes: bytes) -> bool:
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 조건 1: 평균 밝기 (문서는 흰 배경이라 밝음)
    avg_brightness = np.mean(gray)
    if avg_brightness < TH_DOC_BRIGHTNESS_MIN:
        return False

    # 조건 2: 엣지 밀도 (문서는 텍스트 위주라 엣지가 적음)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
    if edge_density > TH_DOC_EDGE_MAX:
        return False

    # 조건 3: 흰 픽셀 비율 (문서는 배경이 흰색)
    white_ratio = np.sum(gray > 200) / (img.shape[0] * img.shape[1])
    if white_ratio < TH_DOC_WHITE_MIN:
        return False

    return True


def _compute_ela(img_bytes: bytes, quality=90, is_png: bool = False) -> float:
    original = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # PNG는 무손실 압축이라 JPEG로 바로 비교하면 포맷 변환 오차가
    # 위조 흔적과 구분되지 않아, 먼저 JPEG로 변환 후 재압축 비교
    if is_png:
        temp_buffer = io.BytesIO()
        original.save(temp_buffer, format="JPEG", quality=95)
        temp_buffer.seek(0)
        original = Image.open(temp_buffer).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    original_arr     = np.array(original,     dtype=np.float32)
    recompressed_arr = np.array(recompressed, dtype=np.float32)
    return float(np.mean(np.abs(original_arr - recompressed_arr) * 10))


def _extract_features(img_bytes: bytes, image_path: str = "") -> list:
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

    # 특징 5: ELA (PNG 여부 전달)
    is_png = image_path.lower().endswith(".png")
    ela_score = _compute_ela(img_bytes, is_png=is_png)

    return [red_ratio, edge_density, avg_brightness, noise_uniformity, ela_score]


# ============================================================
# [근거 생성]
# ============================================================
def _generate_reasons(features: list) -> list:
    red_ratio, edge_density, avg_brightness, noise_uniformity, ela_score = features
    reasons = []

    if ela_score > TH_ELA_HIGH:
        reasons.append("이미지 편집 또는 합성 흔적이 감지되었습니다.")

    if noise_uniformity > TH_NOISE_HIGH:
        reasons.append("이미지 구역별 노이즈 패턴이 불균일합니다.")

    # ── 붉은색 비율 조건 제거 (도장 없는 정상 공문서 오탐 방지) ──────
    # if red_ratio < TH_RED_LOW:
    #     reasons.append("인감 또는 직인으로 추정되는 붉은색 영역이 거의 감지되지 않았습니다.")

    if edge_density < TH_EDGE_LOW:
        reasons.append("문서 내 텍스트 및 선 밀도가 정상 범위보다 낮습니다.")

    # ── 밝기 조건 제거 (위조/정상 모두 220 초과로 구분력 없음) ────────
    # if avg_brightness > TH_BRIGHTNESS_HIGH or avg_brightness < TH_BRIGHTNESS_LOW:
    #     reasons.append("이미지 전체 밝기가 정상 스캔 문서 범위를 벗어났습니다.")

    if not reasons:
        reasons.append("여러 이미지 특징이 복합적으로 정상 문서 패턴에서 벗어났습니다.")

    return reasons


def predict(image_path: str, force: bool = False) -> dict:
    """
    force=False: 문서 판별 후 위조 분석 (기본값)
    force=True:  문서 판별 건너뛰고 바로 위조 분석
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # ── 문서 영역 크롭 (스크린샷 등 불필요한 영역 제거) ──────────────
    # 사각형 윤곽선 탐지로 문서 부분만 추출, 못 찾으면 원본 유지
    img_bytes = _crop_document(img_bytes)
    # ────────────────────────────────────────────────────────────────

    # ── 문서 판별 (force=True면 건너뜀) ──────────────────────────────
    if not force:
        if not _is_document(img_bytes):
            return {
                "document_detected": False,
                "is_forged":         None,
                "result":            "문서 아님",
                "score":             None,
                "reasons":           []
            }
    # ────────────────────────────────────────────────────────────────

    features = _extract_features(img_bytes, image_path=image_path)
    if features is None:
        return {"error": "이미지를 읽을 수 없습니다."}

    # ── 디버그 로그 ───────────────────────────────────────────────────
    # print(f"[DEBUG] red_ratio:        {features[0]:.6f}")
    print(f"[DEBUG] edge_density:     {features[1]:.6f}  (기준: < {TH_EDGE_LOW})")
    # print(f"[DEBUG] avg_brightness:   {features[2]:.2f}")
    print(f"[DEBUG] noise_uniformity: {features[3]:.6f}  (기준: > {TH_NOISE_HIGH})")
    print(f"[DEBUG] ela_score:        {features[4]:.6f}  (기준: > {TH_ELA_HIGH})")
    # ────────────────────────────────────────────────────────────────

    prediction = _model.predict([features])[0]
    score      = float(_model.decision_function([features])[0])
    is_forged  = bool(prediction != 1)
    reasons    = _generate_reasons(features) if is_forged else []

    return {
        "document_detected": True,
        "is_forged":         is_forged,
        "result":            "위조 의심" if is_forged else "정상",
        "score":             round(score, 3),
        "reasons":           reasons
    }