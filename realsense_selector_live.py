# realsense_selector_live_ko.py
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import math
import time
from collections import deque
from PIL import Image, ImageFont, ImageDraw  # 한글 표시용

# ===================== 사용자 설정 =====================
MODEL_PATH = r"selector.pt"     # <- 학습한 가중치 경로로 변경
CONF_THRES = 0.35
IMG_SIZE = 640
DEVICE = 'cpu'              # GPU: 0, CPU: 'cpu'

# 여러 객체가 보일 때: 가장 큰 박스만 표시할지
ONLY_LARGEST = True
# ======================================================

# (1) 각도 유틸 --------------------------------------------------------------
def normalize_angle_deg(angle):
    """-180~+180 범위 각도를 0~180도로 정규화"""
    angle = (angle + 360.0) % 360.0
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def angle_from_kpts(pivot_xy, tip_xy):
    """키포인트(pivot, tip)로부터 각도 계산 (0~180도)"""
    dx = tip_xy[0] - pivot_xy[0]
    dy = tip_xy[1] - pivot_xy[1]
    angle = math.degrees(math.atan2(dy, dx))
    return normalize_angle_deg(angle)

def ema(prev, new, alpha=0.25):
    """지수이동평균으로 각도 스무딩"""
    return new if prev is None else (alpha*new + (1-alpha)*prev)

# (2) 상태 규칙: 요청한 범위 반영 + 겹침구간 타이브레이크 ---------------------
# 구간 (포함 범위)
RANGES = {
    "SAFE":  (110.0, 140.0),
    "SEMI":  (130.0, 160.0),
    "FULL":  (40.0,   80.0),
    "BURST": (5.0,    60.0),
}
# 각 상태의 중심각
CENTERS = {k: (v[0]+v[1])/2.0 for k, v in RANGES.items()}

def in_range(angle, rng):
    lo, hi = rng
    return (angle >= lo) and (angle <= hi)

def classify_angle_with_tiebreak(angle):
    """
    1) angle이 들어가는 모든 상태 후보를 모음
    2) 후보가 여러 개면 중심각과의 거리로 가장 가까운 상태 선택
    3) 후보가 없으면 'RECHECK' 반환
    """
    candidates = [s for s, rng in RANGES.items() if in_range(angle, rng)]
    if not candidates:
        return "RECHECK"
    if len(candidates) == 1:
        return candidates[0]
    # 겹침 구간: 중심각과의 거리 비교
    best, best_dist = None, 1e9
    for s in candidates:
        dist = abs(angle - CENTERS[s])
        if dist < best_dist:
            best_dist = dist
            best = s
    return best

# (3) 한글 폰트 로딩 + Pillow로 텍스트 그리기 -------------------------------
def find_korean_font(preferred_size=22):
    """OS별로 흔한 한글 폰트 경로를 시도해 로드"""
    candidates = [
        # Windows
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        # Ubuntu/Debian (Colab 포함)
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, preferred_size)
        except Exception:
            continue
    return None  # 폰트를 못 찾으면 None

_KO_FONT_SMALL  = find_korean_font(20)
_KO_FONT_MEDIUM = find_korean_font(24)

def draw_korean_text(img_bgr, text, org_xy, font=None, color_bgr=(255,255,255)):
    """
    Pillow로 유니코드(한글) 텍스트 그리기
    img_bgr: OpenCV BGR 이미지
    org_xy:  (x, y) 좌표
    color_bgr: (B, G, R)
    """
    if font is None:
        font = _KO_FONT_MEDIUM
    if font is None:
        # 폰트를 못 찾은 경우: 마지막 수단으로 OpenCV로 ASCII만 출력
        cv2.putText(img_bgr, text, org_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text(org_xy, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# (4) 시각화: 도형은 OpenCV, 텍스트는 Pillow ---------------------------------
def draw_one(vis, x1, y1, x2, y2, pivot, tip, angle, state):
    """
    상태 메시지:
      - SAFE: 'SAFE (안전한 상태)'
      - RECHECK: '조정간 상태 재확인'
      - 그 외(SEMI/FULL/BURST): '{STATE} (격발 가능 상태)'
    """
    if state == "SAFE":
        state_msg = "SAFE (안전한 상태)"
        color = (0, 200, 0)
    elif state == "RECHECK":
        state_msg = "조정간 상태 재확인"
        color = (0, 165, 255)  # 주황
    else:
        state_msg = f"{state} (격발 가능 상태)"
        color = (0, 0, 255)

    # 도형은 OpenCV로
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.circle(vis, (int(pivot[0]), int(pivot[1])), 5, (255, 0, 0), -1)  # pivot: 파랑
    cv2.circle(vis, (int(tip[0]),   int(tip[1])),   5, (0, 0, 255), -1)  # tip: 빨강
    cv2.line(vis, (int(pivot[0]), int(pivot[1])), (int(tip[0]), int(tip[1])), (0, 255, 255), 2)

    # 텍스트는 Pillow로 (한글 깨짐 방지)
    top_y = max(22, y1 - 12)
    vis = draw_korean_text(vis, f"lever | angle: {angle:.1f} deg", (x1, top_y),
                           font=_KO_FONT_SMALL, color_bgr=(255, 255, 255))
    vis = draw_korean_text(vis, state_msg, (x1, top_y + 24),
                           font=_KO_FONT_MEDIUM, color_bgr=color)
    return vis

# (5) 메인 루프 --------------------------------------------------------------
def main():
    print("[INFO] Loading model...")
    model = YOLO(MODEL_PATH)

    print("[INFO] Starting RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    smooth_angle = None
    last_save_idx = 0

    try:
        t0 = time.time(); frames_cnt = 0
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            vis = frame.copy()

            # 추론
            results = model.predict(
                source=frame,
                conf=CONF_THRES,
                imgsz=IMG_SIZE,
                device=DEVICE,
                verbose=False
            )

            best_area = -1
            best_pack = None
            detections = []

            for r in results:
                boxes = r.boxes
                kpts = r.keypoints
                if boxes is None or kpts is None or len(boxes) == 0:
                    continue

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                    area = (x2 - x1) * (y2 - y1)
                    kxy = kpts.xy[i].cpu().numpy()  # (2,2): pivot, tip
                    pivot, tip = kxy[0], kxy[1]
                    detections.append((x1, y1, x2, y2, area, pivot, tip))
                    if area > best_area:
                        best_area = area
                        best_pack = (x1, y1, x2, y2, area, pivot, tip)

            if detections:
                if ONLY_LARGEST:
                    x1, y1, x2, y2, area, pivot, tip = best_pack
                    angle = angle_from_kpts(pivot, tip)
                    smooth_angle = ema(smooth_angle, angle, alpha=0.25)
                    state = classify_angle_with_tiebreak(smooth_angle)
                    vis = draw_one(vis, x1, y1, x2, y2, pivot, tip, smooth_angle, state)
                else:
                    for (x1, y1, x2, y2, area, pivot, tip) in detections:
                        angle = angle_from_kpts(pivot, tip)
                        state = classify_angle_with_tiebreak(angle)
                        vis = draw_one(vis, x1, y1, x2, y2, pivot, tip, angle, state)
            else:
                # 검출 실패 메시지도 한글로
                vis = draw_korean_text(vis, "조정간이 탐지되지 않았습니다.", (10, 30),
                                       font=_KO_FONT_MEDIUM, color_bgr=(0, 0, 255))

            # FPS 표시(영문으로 유지)
            frames_cnt += 1
            if frames_cnt >= 10:
                t1 = time.time()
                fps = frames_cnt / (t1 - t0)
                t0 = t1
                frames_cnt = 0
            else:
                fps = None

            if fps is not None:
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, vis.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("RealSense YOLOv8 Pose - Selector Lever (KO)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                out_name = f"selector_frame_{last_save_idx:04d}.png"
                cv2.imwrite(out_name, vis)
                print(f"[INFO] Saved: {out_name}")
                last_save_idx += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
