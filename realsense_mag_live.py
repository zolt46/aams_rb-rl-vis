# mag_top_detect_sideways_ko.py
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
from PIL import Image, ImageFont, ImageDraw
from collections import deque

# ===================== 사용자 설정 =====================
MODEL_PATH = r"mag.pt"    # <- YOLOv8 detect 가중치 (2클래스: left_head, right_head)
CONF_THRES = 0.35
IMG_SIZE = 640
DEVICE = 'cpu'                   # GPU: 0, CPU: 'cpu'

THRESH_X_DIFF = 10.0             # 두 클래스의 중심 x 차이가 너무 작으면 재확인
THRESH_BOX_CONF = 0.50           # 박스 신뢰도 낮으면 해당 박스는 무시
N_CONSENSUS = 4                  # 같은 판정 N프레임 연속일 때만 확정
SHOW_RAW_CAMERA = True           # 원본 카메라 영상 창 표시

# (선택) 동일 클래스가 여러개 잡힐 때 사용: 가장 높은 conf 1개만 쓰기(True) / 평균 cx 사용(False)
USE_TOP1_PER_CLASS = True
# ======================================================

# -------- 한글 폰트 로딩 / 텍스트 유틸 --------
def find_korean_font(preferred_size=22):
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, preferred_size)
        except Exception:
            continue
    return None

FONT_S = find_korean_font(20)
FONT_M = find_korean_font(24)

def draw_korean_text(img_bgr, text, org_xy, font=None, color_bgr=(255,255,255)):
    if font is None: font = FONT_M
    if font is None:
        cv2.putText(img_bgr, text, org_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        return img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text(org_xy, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# -------- 판정/버퍼 --------
recent_labels = deque(maxlen=N_CONSENSUS)

def center_of_box(x1, y1, x2, y2):
    return (0.5*(x1+x2), 0.5*(y1+y2))

def decide_side_by_cx(cx_left, cx_right, x_diff_th=THRESH_X_DIFF):
    """
    옆으로 돌린 상태: x가 더 큰 쪽이 '상'
    - cx_left > cx_right  -> 좌상탄
    - cx_right > cx_left  -> 우상탄
    - |cx_left - cx_right| < x_diff_th -> 재확인
    """
    dx = cx_left - cx_right
    if abs(dx) < x_diff_th:
        return "재확인", dx
    return ("좌상탄", dx) if dx > 0 else ("우상탄", dx)

def pick_center_for_class(boxes, confs, cls_ids, target_cls, conf_th=THRESH_BOX_CONF, use_top1=True):
    """
    주어진 detection들에서 특정 클래스(target_cls)의 중심 cx를 뽑는다.
    - conf < conf_th는 무시
    - use_top1=True: 가장 conf 높은 1개
      use_top1=False: 모든 박스의 cx 평균
    반환: (ok, cx, chosen_indices)
    """
    idxs = [i for i, c in enumerate(cls_ids) if c == target_cls and confs[i] >= conf_th]
    if not idxs:
        return False, None, []
    if use_top1:
        best_i = max(idxs, key=lambda i: confs[i])
        x1,y1,x2,y2 = boxes[best_i]
        cx,_ = center_of_box(x1,y1,x2,y2)
        return True, cx, [best_i]
    else:
        cxs = []
        chosen = []
        for i in idxs:
            x1,y1,x2,y2 = boxes[i]
            cx,_ = center_of_box(x1,y1,x2,y2)
            cxs.append(cx); chosen.append(i)
        return True, float(np.mean(cxs)), chosen

def main():
    print("[INFO] Loading detect model...")
    model = YOLO(MODEL_PATH)

    print("[INFO] Starting RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        t0 = time.time(); frames_cnt = 0
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            vis = frame.copy()

            if SHOW_RAW_CAMERA:
                cv2.imshow("Camera (Raw)", frame)

            results = model.predict(
                frame, conf=CONF_THRES, imgsz=IMG_SIZE, device=DEVICE, verbose=False
            )

            label_str = "재확인"
            color = (0,165,255)

            any_det = False
            for r in results:
                boxes_obj = r.boxes
                names = r.names if hasattr(r, "names") else {}
                if boxes_obj is None or len(boxes_obj) == 0:
                    continue
                any_det = True

                # 박스/클래스/점수 배열로 꺼내기
                xyxy = boxes_obj.xyxy.cpu().numpy()
                conf = boxes_obj.conf.cpu().numpy() if hasattr(boxes_obj, "conf") else np.ones((len(boxes_obj),), dtype=np.float32)
                cls  = boxes_obj.cls.cpu().numpy().astype(int) if hasattr(boxes_obj, "cls") else -np.ones((len(boxes_obj),), dtype=int)

                # 전부 그리기
                for i in range(len(xyxy)):
                    x1,y1,x2,y2 = map(int, xyxy[i])
                    cls_id = int(cls[i])
                    box_conf = float(conf[i])
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,255), 2)
                    vis = draw_korean_text(vis, f"{label}  conf:{box_conf:.2f}", (x1, max(22, y1-12)),
                                           font=FONT_S, color_bgr=(0,200,255))

                # 좌/우 클래스 중심 cx 구하기
                # 클래스 인덱스 가정: 0='left_head', 1='right_head'  (names 순서 확인 필수!)
                # names 출력해서 실제 클래스 인덱스를 꼭 확인하세요.
                # 필요하면 아래 두 줄을 names 문자열 비교 방식으로 수정 가능.
                okL, cxL, idxL = pick_center_for_class(xyxy, conf, cls, target_cls=0,
                                                       conf_th=THRESH_BOX_CONF, use_top1=USE_TOP1_PER_CLASS)
                okR, cxR, idxR = pick_center_for_class(xyxy, conf, cls, target_cls=1,
                                                       conf_th=THRESH_BOX_CONF, use_top1=USE_TOP1_PER_CLASS)

                # 중심점 위치 시각화(선택)
                for i in (idxL + idxR):
                    x1,y1,x2,y2 = map(int, xyxy[i])
                    cx, cy = center_of_box(x1,y1,x2,y2)
                    cv2.circle(vis, (int(cx), int(cy)), 4, (0,255,0), -1)

                if okL and okR:
                    cur, dx = decide_side_by_cx(cxL, cxR)
                    if cur != "재확인":
                        recent_labels.append(cur)
                        if len(recent_labels) == N_CONSENSUS and len(set(recent_labels)) == 1:
                            label_str = f"{cur} (확정)"
                            color = (0,200,0)
                        else:
                            label_str = f"{cur}? (확인중)"
                            color = (255,255,0)
                    else:
                        recent_labels.clear()
                        label_str = "재확인"
                        color = (0,165,255)

                    # dx 디버그 출력
                    vis = draw_korean_text(vis, f"Δx={dx:.1f} (x_right - x_left의 부호 반전)", (10, 30),
                                           font=FONT_S, color_bgr=(200,200,200))
                else:
                    recent_labels.clear()
                    # 왜 재확인인지 이유 표시
                    reason = []
                    if not okL: reason.append("left 부족/저신뢰")
                    if not okR: reason.append("right 부족/저신뢰")
                    vis = draw_korean_text(vis, " / ".join(reason) if reason else "재확인", (10, 30),
                                           font=FONT_S, color_bgr=(0,165,255))

            if not any_det:
                vis = draw_korean_text(vis, "감지 없음 (재확인)", (10, 30), font=FONT_M, color_bgr=(0,0,255))

            # 최종 결과 텍스트
            vis = draw_korean_text(vis, f"결과: {label_str}", (10, 60), font=FONT_M, color_bgr=color)

            # FPS
            frames_cnt += 1
            if frames_cnt >= 10:
                t1 = time.time()
                fps = frames_cnt / (t1 - t0)
                t0 = t1
                frames_cnt = 0
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, vis.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Mag Top - Detect (Sideways, KO)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fn = f"mag_top_{int(time.time())}.png"
                cv2.imwrite(fn, vis)
                print("[INFO] saved:", fn)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
