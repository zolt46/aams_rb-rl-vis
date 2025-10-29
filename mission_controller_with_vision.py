# -*- coding: utf-8 -*-
"""
mission_controller_with_vision.py
- 불입(IN) 및 불출(OUT) 프로세스 완전 구현
- 리니어 레일 제어 통합
- 비전 검사 통합:
  1) QR 코드 인식 및 검증
  2) 조정간 안전 상태 확인 (SAFE만 통과)
  3) 탄창 방향 확인 (우상탄만 통과)
"""

import argparse
import json
import queue
import re
import sys
import time
import threading
import uuid
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import math
from collections import deque
from PIL import Image, ImageFont, ImageDraw
from pyzbar.pyzbar import decode as pyzbar_decode
from typing import Dict, Any, List, Iterable, Optional, Callable, Tuple, Union
from contextlib import contextmanager
from datetime import datetime
from robot_sdk_v13_rail_extended import BridgeClient

TEACH_FILE = "final_asset.json"
KEEPALIVE_INTERVAL = 5.0

# 레일 설정
RAIL_MAX_STROKE = 800.0
RAIL_FORWARD_SPEED = 200.0
RAIL_BACKWARD_SPEED = 200.0
RAIL_HOME_SPEED = 200.0

# 비전 설정
SELECTOR_MODEL_PATH = r"selector.pt"
MAG_MODEL_PATH = r"mag.pt"
VISION_CONF_THRES = 0.35
VISION_IMG_SIZE = 640
VISION_DEVICE = 'cpu'

# 조정간 상태 범위 (각도 기준)
SELECTOR_RANGES = {
    "SAFE":  (110.0, 140.0),
    "SEMI":  (130.0, 160.0),
    "FULL":  (40.0,   80.0),
    "BURST": (5.0,    60.0),
}
SELECTOR_CENTERS = {k: (v[0]+v[1])/2.0 for k, v in SELECTOR_RANGES.items()}

# 탄창 검사 설정
MAG_N_CONSENSUS = 4  # 같은 판정 N프레임 연속
MAG_THRESH_X_DIFF = 10.0
MAG_THRESH_BOX_CONF = 0.50

# ----------------------------- 브리지 인터페이스 -----------------------------
class BridgeAbort(Exception):
    """외부 인터랙션으로 집행이 중단된 경우"""


class BridgeInterface:
    def __init__(self, enabled: bool = False, context: dict | None = None):
        self.enabled = bool(enabled)
        self.context = dict(context or {})
        self._waiters: dict[str, queue.Queue] = {}
        self._early: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._listener: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not self.enabled:
            return
        if self._listener and self._listener.is_alive():
            return
        self._running = True
        self._listener = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._running = False

    def emit(self, event: str, **payload) -> None:
        if not self.enabled:
            return
        data = {"event": event, **payload}
        request_id = self.context.get("request_id") or self.context.get("requestId")
        if request_id and "requestId" not in data:
            data["requestId"] = request_id
        mission_label = self.context.get("mission_label")
        if mission_label and "missionLabel" not in data:
            data.setdefault("meta", {}).setdefault("missionLabel", mission_label)
        try:
            print(json.dumps(data, ensure_ascii=False), flush=True)
        except Exception:
            pass

    def wait_for(self, stage: str, message: str, *, allow_cancel: bool = False, meta: dict | None = None) -> dict:
        if not self.enabled:
            return {}
        token = uuid.uuid4().hex
        queue_obj = self._register_waiter(token)
        payload = {
            "stage": stage,
            "message": message,
            "token": token,
            "allowCancel": allow_cancel
        }
        if meta:
            payload["meta"] = meta
        self.emit("await_user", **payload)
        try:
            while self._running:
                try:
                    command = queue_obj.get(timeout=0.1)
                except queue.Empty:
                    continue
                cmd = str(command.get("command") or "").strip().lower()
                if cmd in {"resume", "continue", "ok", "next"}:
                    self.emit("await_user_done", stage=stage, message=message, token=token, command=cmd)
                    return command
                if cmd in {"abort", "cancel", "stop"}:
                    self.emit(
                        "await_user_done",
                        stage=stage,
                        message="사용자 중단",
                        token=token,
                        command=cmd,
                        aborted=True
                    )
                    raise BridgeAbort(command.get("reason") or "aborted")
        finally:
            self._unregister_waiter(token)
        raise BridgeAbort("bridge_stopped")

    def _register_waiter(self, token: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._waiters[token] = q
            pending = self._early.pop(token, [])
        for entry in pending:
            q.put(entry)
        return q

    def _unregister_waiter(self, token: str) -> None:
        with self._lock:
            self._waiters.pop(token, None)

    def _listen_loop(self) -> None:
        while self._running:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                if not self._running:
                    break
                time.sleep(0.05)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = data.get("token")
            if not token:
                continue
            with self._lock:
                if token in self._waiters:
                    self._waiters[token].put(data)
                else:
                    self._early.setdefault(token, []).append(data)


def resolve_mission_number(raw_value: Optional[Union[int, str]], fallback_label: Optional[str] = None) -> int:
    candidates = []
    if raw_value is not None:
        candidates.append(raw_value)
    if fallback_label is not None:
        candidates.append(fallback_label)
    for candidate in candidates:
        try:
            value = int(candidate)
            if value in (1, 2):
                return value
        except (TypeError, ValueError):
            digits = ''.join(ch for ch in str(candidate) if ch.isdigit())
            if digits:
                try:
                    value = int(digits)
                    if value in (1, 2):
                        return value
                except ValueError:
                    continue
    return 1

# ----------------------------- JSON 로드 -----------------------------
JOINT_KEY_PATTERNS = (
    "Joint{idx}(deg)",
    "Joint{idx}",
    "Joint{idx}_deg",
    "Joint{idx}Deg",
    "Jnt{idx}",
    "J{idx}",
)


def _coalesce_joint_value(rec: dict, idx: int) -> float:
    for key_tpl in JOINT_KEY_PATTERNS:
        key = key_tpl.format(idx=idx)
        if key in rec:
            val = rec.get(key)
            if val is None:
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 0.0

def _extract_records(obj) -> List[dict]:
    if isinstance(obj, dict):
        if "teach_data" in obj and isinstance(obj["teach_data"], list):
            return obj["teach_data"]
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        raise ValueError("JSON dict 구조에서 teach_data 리스트를 찾지 못했습니다.")
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            return obj
        raise ValueError("JSON이 list지만 dict 레코드가 아닙니다.")
    else:
        raise ValueError("알 수 없는 JSON 루트 타입입니다. dict 또는 list여야 합니다.")

def _parse_dt(s: Optional[str]) -> datetime:
    if not s:
        return datetime.min
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return datetime.min

def load_positions_by_label(fn: str, wanted: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    try:
        with open(fn, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        # teach JSON에 문법 오류가 있는 경우 보다 이해하기 쉬운 메시지 제공
        raise ValueError(f"포지션 JSON 구문 오류({fn}): {e}") from e
    records = _extract_records(raw)

    buckets: Dict[str, List[dict]] = {}
    for rec in records:
        name = rec.get("Position Name") or rec.get("PositionName")
        if not name:
            continue
        key = name.strip()
        if wanted and key not in wanted:
            continue
        buckets.setdefault(key, []).append(rec)

    positions: Dict[str, Dict[str, Any]] = {}
    for label, recs in buckets.items():
        latest = max(recs, key=lambda r: _parse_dt(r.get("DateTime")))
        normalized = dict(latest)
        for idx in range(1, 7):
            normalized[f"Joint{idx}(deg)"] = _coalesce_joint_value(latest, idx)
        positions[label] = normalized

    if not positions:
        raise ValueError("JSON에서 유효한 라벨 포지션을 찾지 못했습니다.")
    return positions

# ----------------------------- 비전 유틸리티 -----------------------------
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

# ----------------------------- 조정간 비전 검사 -----------------------------
def normalize_angle_deg(angle):
    angle = (angle + 360.0) % 360.0
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def angle_from_kpts(pivot_xy, tip_xy):
    dx = tip_xy[0] - pivot_xy[0]
    dy = tip_xy[1] - pivot_xy[1]
    angle = math.degrees(math.atan2(dy, dx))
    return normalize_angle_deg(angle)

def in_range(angle, rng):
    lo, hi = rng
    return (angle >= lo) and (angle <= hi)

def classify_selector_angle(angle):
    candidates = [s for s, rng in SELECTOR_RANGES.items() if in_range(angle, rng)]
    if not candidates:
        return "RECHECK"
    if len(candidates) == 1:
        return candidates[0]
    best, best_dist = None, 1e9
    for s in candidates:
        dist = abs(angle - SELECTOR_CENTERS[s])
        if dist < best_dist:
            best_dist = dist
            best = s
    return best

def check_selector_safe(model, frame, max_attempts=30, display=True, confirm_frames=3) -> Tuple[bool, str]:
    """
    조정간이 SAFE 상태인지 확인
    Returns: (is_safe, message)
    """
    print("[VISION] 조정간 안전 상태 확인 중...")

    not_safe_count = 0
    last_state_message = "조정간 상태 확인 중"
    for attempt in range(max_attempts):
        vis = frame.copy() if display else None
        
        results = model.predict(
            source=frame,
            conf=VISION_CONF_THRES,
            imgsz=VISION_IMG_SIZE,
            device=VISION_DEVICE,
            verbose=False
        )
        
        detected = False
        is_safe = False
        current_state = "RECHECK"
        
        for r in results:
            boxes = r.boxes
            kpts = r.keypoints
            if boxes is None or kpts is None or len(boxes) == 0:
                continue
                
            detected = True
            x1, y1, x2, y2 = map(int, boxes.xyxy[0].cpu().numpy())
            kxy = kpts.xy[0].cpu().numpy()
            pivot, tip = kxy[0], kxy[1]
            
            angle = angle_from_kpts(pivot, tip)
            current_state = classify_selector_angle(angle)
            is_safe = (current_state == "SAFE")
            
            if display:
                color = (0, 200, 0) if is_safe else (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.circle(vis, (int(pivot[0]), int(pivot[1])), 5, (255, 0, 0), -1)
                cv2.circle(vis, (int(tip[0]), int(tip[1])), 5, (0, 0, 255), -1)
                cv2.line(vis, (int(pivot[0]), int(pivot[1])), 
                        (int(tip[0]), int(tip[1])), (0, 255, 255), 2)
                
                state_msg = f"조정간: {current_state} | 각도: {angle:.1f}도"
                vis = draw_korean_text(vis, state_msg, (x1, max(22, y1-12)),
                                     font=FONT_S, color_bgr=color)
            
            break
        
        if not detected:
            if display:
                vis = draw_korean_text(vis, "조정간 미감지", (10, 30),
                                     font=FONT_M, color_bgr=(0, 0, 255))
        
        if display and vis is not None:
            cv2.imshow("Selector Check", vis)
            cv2.waitKey(50)
        
        if detected:
            if is_safe:
                if display:
                    cv2.destroyWindow("Selector Check")
                return True, f"조정간 SAFE 확인 완료 (각도: {angle:.1f}도)"
            else:
                not_safe_count += 1
                last_state_message = f"조정간이 SAFE가 아닙니다. 현재 상태: {current_state} (각도: {angle:.1f}도)"
                if not_safe_count >= max(1, confirm_frames):
                    if display:
                        cv2.destroyWindow("Selector Check")
                    return False, last_state_message
        else:
            not_safe_count = 0

        time.sleep(0.05)

    if display:
        cv2.destroyWindow("Selector Check")

    if not_safe_count:
        return False, last_state_message
    return False, "조정간을 감지하지 못했습니다."

# ----------------------------- 탄창 비전 검사 -----------------------------
def center_of_box(x1, y1, x2, y2):
    return (0.5*(x1+x2), 0.5*(y1+y2))

def decide_mag_side_by_cx(cx_left, cx_right, x_diff_th=MAG_THRESH_X_DIFF):
    dx = cx_left - cx_right
    if abs(dx) < x_diff_th:
        return "재확인", dx
    return ("좌상탄", dx) if dx > 0 else ("우상탄", dx)

def pick_center_for_class(boxes, confs, cls_ids, target_cls, conf_th=MAG_THRESH_BOX_CONF):
    idxs = [i for i, c in enumerate(cls_ids) if c == target_cls and confs[i] >= conf_th]
    if not idxs:
        return False, None, []
    best_i = max(idxs, key=lambda i: confs[i])
    x1,y1,x2,y2 = boxes[best_i]
    cx,_ = center_of_box(x1,y1,x2,y2)
    return True, cx, [best_i]

def check_mag_orientation(model, frame, max_attempts=100, display=True) -> Tuple[bool, str]:
    """
    탄창이 우상탄인지 확인
    Returns: (is_correct, message)
    """
    print("[VISION] 탄창 방향 확인 중...")
    recent_labels = deque(maxlen=MAG_N_CONSENSUS)
    
    for attempt in range(max_attempts):
        vis = frame.copy() if display else None
        
        results = model.predict(
            frame, conf=VISION_CONF_THRES, imgsz=VISION_IMG_SIZE, 
            device=VISION_DEVICE, verbose=False
        )
        
        label_str = "재확인"
        color = (0,165,255)
        detected = False
        
        for r in results:
            boxes_obj = r.boxes
            names = r.names if hasattr(r, "names") else {}
            if boxes_obj is None or len(boxes_obj) == 0:
                continue
            
            detected = True
            xyxy = boxes_obj.xyxy.cpu().numpy()
            conf = boxes_obj.conf.cpu().numpy() if hasattr(boxes_obj, "conf") else np.ones((len(boxes_obj),), dtype=np.float32)
            cls = boxes_obj.cls.cpu().numpy().astype(int) if hasattr(boxes_obj, "cls") else -np.ones((len(boxes_obj),), dtype=int)
            
            if display:
                for i in range(len(xyxy)):
                    x1,y1,x2,y2 = map(int, xyxy[i])
                    cls_id = int(cls[i])
                    box_conf = float(conf[i])
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,255), 2)
                    vis = draw_korean_text(vis, f"{label} {box_conf:.2f}", (x1, max(22, y1-12)),
                                         font=FONT_S, color_bgr=(0,200,255))
            
            okL, cxL, idxL = pick_center_for_class(xyxy, conf, cls, target_cls=0)
            okR, cxR, idxR = pick_center_for_class(xyxy, conf, cls, target_cls=1)
            
            if display:
                for i in (idxL + idxR):
                    x1,y1,x2,y2 = map(int, xyxy[i])
                    cx, cy = center_of_box(x1,y1,x2,y2)
                    cv2.circle(vis, (int(cx), int(cy)), 4, (0,255,0), -1)
            
            if okL and okR:
                cur, dx = decide_mag_side_by_cx(cxL, cxR)
                if cur != "재확인":
                    recent_labels.append(cur)
                    if len(recent_labels) == MAG_N_CONSENSUS and len(set(recent_labels)) == 1:
                        label_str = f"{cur} (확정)"
                        color = (0,200,0) if cur == "우상탄" else (0,0,255)
                    else:
                        label_str = f"{cur}? (확인중)"
                        color = (255,255,0)
                else:
                    recent_labels.clear()
            else:
                recent_labels.clear()
        
        if not detected:
            if display:
                vis = draw_korean_text(vis, "탄창 미감지", (10, 30),
                                     font=FONT_M, color_bgr=(0,0,255))
        
        if display and vis is not None:
            vis = draw_korean_text(vis, f"결과: {label_str}", (10, 60),
                                 font=FONT_M, color_bgr=color)
            cv2.imshow("Magazine Check", vis)
            cv2.waitKey(50)
        
        if len(recent_labels) == MAG_N_CONSENSUS and len(set(recent_labels)) == 1:
            final_result = recent_labels[0]
            if display:
                cv2.destroyWindow("Magazine Check")
            
            if final_result == "우상탄":
                return True, "탄창 방향 확인: 우상탄 (정상)"
            else:
                return False, f"탄창 방향 오류: {final_result} 감지됨"
        
        time.sleep(0.1)
    
    if display:
        cv2.destroyWindow("Magazine Check")
    
    return False, "탄창 방향 확인 실패: 시간 초과"

# ----------------------------- QR 코드 검사 -----------------------------
def check_qr_code(frame, expected_value: str, max_attempts=30, display=True) -> Tuple[bool, str]:
    """
    QR 코드 인식 및 검증
    Returns: (is_valid, message)
    """
    print(f"[VISION] QR 코드 확인 중... (기대값: {expected_value})")
    
    for attempt in range(max_attempts):
        vis = frame.copy() if display else None
        
        # QR 코드 디코딩
        decoded_objects = pyzbar_decode(frame)
        
        if decoded_objects:
            for obj in decoded_objects:
                qr_data = obj.data.decode('utf-8')
                qr_type = obj.type
                
                # QR 코드 영역 표시
                if display:
                    points = obj.polygon
                    if len(points) == 4:
                        pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
                        color = (0, 255, 0) if qr_data == expected_value else (0, 0, 255)
                        cv2.polylines(vis, [pts], True, color, 3)
                        
                        x, y = obj.rect.left, obj.rect.top
                        vis = draw_korean_text(vis, f"QR: {qr_data}", (x, max(22, y-12)),
                                             font=FONT_S, color_bgr=color)
                
                # 검증
                if qr_data == expected_value:
                    if display:
                        cv2.destroyWindow("QR Check")
                    return True, f"QR 코드 확인 완료: {qr_data}"
                else:
                    if display:
                        vis = draw_korean_text(vis, 
                                             f"QR 불일치! 기대: {expected_value}, 실제: {qr_data}",
                                             (10, 60), font=FONT_M, color_bgr=(0, 0, 255))
        else:
            if display:
                vis = draw_korean_text(vis, "QR 코드 미감지", (10, 30),
                                     font=FONT_M, color_bgr=(0, 165, 255))
        
        if display and vis is not None:
            cv2.imshow("QR Check", vis)
            cv2.waitKey(50)
        
        time.sleep(0.1)
    
    if display:
        cv2.destroyWindow("QR Check")
    
    return False, "QR 코드 확인 실패: 시간 초과 또는 불일치"

# ----------------------------- 비전 컨트롤러 -----------------------------
class VisionController:
    def __init__(self):
        print("[VISION] 비전 시스템 초기화...")
        self.pipeline = None
        self.selector_model = None
        self.mag_model = None
        
    def initialize(self):
        """RealSense 및 YOLO 모델 초기화"""
        try:
            # RealSense 초기화
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.pipeline.start(config)
            print("[VISION] RealSense 카메라 초기화 완료")
            
            # YOLO 모델 로드
            self.selector_model = YOLO(SELECTOR_MODEL_PATH)
            print(f"[VISION] 조정간 모델 로드 완료: {SELECTOR_MODEL_PATH}")
            
            self.mag_model = YOLO(MAG_MODEL_PATH)
            print(f"[VISION] 탄창 모델 로드 완료: {MAG_MODEL_PATH}")
            
        except Exception as e:
            print(f"[VISION ERROR] 초기화 실패: {e}")
            raise
    
    def get_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 획득"""
        if not self.pipeline:
            return None
        
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        
        return np.asanyarray(color_frame.get_data())
    
    def check_selector_safe(self, display=True) -> Tuple[bool, str]:
        """조정간 안전 상태 확인"""
        if not self.selector_model:
            return False, "조정간 모델이 로드되지 않았습니다"
        
        frame = self.get_frame()
        if frame is None:
            return False, "프레임을 가져올 수 없습니다"
        
        return check_selector_safe(self.selector_model, frame, display=display)
    
    def check_mag_orientation(self, display=True) -> Tuple[bool, str]:
        """탄창 방향 확인"""
        if not self.mag_model:
            return False, "탄창 모델이 로드되지 않았습니다"
        
        frame = self.get_frame()
        if frame is None:
            return False, "프레임을 가져올 수 없습니다"
        
        return check_mag_orientation(self.mag_model, frame, display=display)
    
    def check_qr_code(self, expected_value: str, display=True) -> Tuple[bool, str]:
        """QR 코드 확인"""
        frame = self.get_frame()
        if frame is None:
            return False, "프레임을 가져올 수 없습니다"
        
        return check_qr_code(frame, expected_value, display=display)
    
    def cleanup(self):
        """리소스 정리"""
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("[VISION] RealSense 카메라 종료")
            except Exception:
                pass
        cv2.destroyAllWindows()

# ----------------------------- 스텝 자료구조 -----------------------------
class Step:
    def __init__(self, title: str, description: str,
                 run_fn: Callable[[], None],
                 preview_labels: Optional[List[str]] = None):
        self.title = title
        self.description = description
        self.run_fn = run_fn
        self.preview_labels = preview_labels or []

# ----------------------------- 미션 컨트롤러 -----------------------------
class MissionController:
    REQUIRED_LABELS = [
        # rifle1
        "pick_rifle1", "grip_rifle1", "up_rifle1",
        # rifle2
        "pick_rifle2", "grip_rifle2", "up1_rifle2", "up2_rifle2", "up3_rifle2", "up4_rifle2",
        # rail
        "up1_rifle", "up2_rifle",
        "on_rail1", "on_rail2", "on_rail3", "on_rail4", "on_rail5", "on_rail6", "fallback_rail",
        # selector
        "go_selector1", "go_selector2", "go_selector3",
        "spin_selector_semi", "out_selector1", "out_selector2",
        "go_selector_from_fire1", "go_selector_from_fire2",
        "spin_selector_safe", "out_selector_safe_end1", "out_selector_safe_end2",
        # cocking & fire
        "go_cocking1", "go_cocking2", "go_cocking3", "cocking",
        "out_cocking1", "out_cocking2",
        "go_fire1", "go_fire2", "fire", "out_fire1", "out_fire2", "out_fire2selector",
        # mag & mag rail
        "go2mag1", "grip_mag1", "pickup_mag1",
        "go_mag2", "grip_mag2", "pickup_mag2", "out_mag2",
        "go_mag_rail1", "go_mag_rail2", "go_mag_rail3_vision",
        "go_mag_rail4", "go_mag_rail5", "go_mag_rail6",
        "pickdown_rail", "out_mag_rail1", "out_mag_rail2", "out_mag_rail3",
        # 불입 전용 포지션들
        "go_pick_mag_return", "o_mag2vision_return1", "go_mag2vision_return2",
        "check_vision_mag_return", "vision2mag_return",
        "place_mag1_return1", "place_mag1_return2",
        "out_mag1_return1", "out_mag1_return2", "out_mag1_return3",
        "pick_mag_return", "go_pick_mag_return",
        "goto_place_return1", "goto_place_return2",
        "go_grip_rifle1_return1 [0]", "go_grip_rifle1_return2",
        "grip_rifle1_return", "out_grip_rifle1_return1", "out_grip_rifle1_return2",
        "out_grip_rifle1_return3", "go_place_rifle1_return4",
        "place_rifle1_return1", "place_rifle1_return2", "down_rifle1_return",
        "out_return", "out_grip_rifle2_return2", "out_grip_rifle2_return4",
        "2_pick_rifle2 [7]",
        # home
        "home_rdy",
    ]

    LABEL_ALIASES = {
        "grip_mag_return": "go_pick_mag_return",
        "pickup_mag_return": "pick_mag_return",
        "go_mag2vision1_return": "o_mag2vision_return1",
        "go_mag2vision2_return": "go_mag2vision_return2",
        "go_vision2mag_return": "vision2mag_return",
        "place_mag1_1_return": "place_mag1_return1",
        "place_mag1_2_return": "place_mag1_return2",
        "out_mag1_1_return": "out_mag1_return1",
        "out_mag1_2_return": "out_mag1_return2",
        "go_mag2rifle_return": "goto_place_return1",
        "rifle_pickgo1_return": "go_grip_rifle1_return1 [0]",
        "rifle_pickgo2_return": "go_grip_rifle1_return2",
        "rifle_pickgo3_return": "grip_rifle1_return",
        "rifle_pickgo4_return": "out_grip_rifle1_return1",
        "rifle1_pickgo1_return": "out_grip_rifle1_return2",
        "rifle1_pickgo2_return": "out_grip_rifle1_return3",
        "rifle1_pickgo3_return": "go_place_rifle1_return4",
        "rifle1_pickgo4_return": "place_rifle1_return1",
        "rifle1_pickgo5_return": "place_rifle1_return2",
        "rifle1_pickgo6_return": "down_rifle1_return",
        "rifle1_pickgo7_return": "out_return",
        "go_mag2rifle_return": "2_pick_rifle2 [7]",
        "rifle2_pickgo1_return": "out_grip_rifle2_return2",
        "rifle2_pickgo3_return": "out_grip_rifle2_return4",
        "rifle2_pickgo4_return": "goto_place_return1",
        "rifle2_pickgo5_return": "goto_place_return2",
    }


    def __init__(self, mission_number: int, direction: str, with_mag: bool,
                 expected_qr: str = "1",
                 bridge_host: str = "192.168.1.23",
                 enable_vision: bool = True,
                 enable_qr_check: Optional[bool] = None,
                 enable_selector_check: Optional[bool] = None,
                 enable_mag_check: Optional[bool] = None,
                 *,
                 bridge: Optional[BridgeInterface] = None,
                 auto_mode: bool = False,
                 mission_label: Optional[str] = None,
                 request_id: Optional[str] = None,
                 site: Optional[str] = None):
        mission_number = resolve_mission_number(mission_number, mission_label)
        if mission_number not in (1, 2):
            raise ValueError("mission_number must be 1 or 2")
        if direction not in ("in", "out"):
            raise ValueError("direction must be 'in' or 'out'")

        self.mission_number = mission_number
        self.direction = direction
        self.with_mag = with_mag
        self.expected_qr = expected_qr
        self.bridge_host = bridge_host
        self.enable_vision = enable_vision
        self.enable_qr_check = enable_vision if enable_qr_check is None else enable_qr_check
        self.enable_selector_check = enable_vision if enable_selector_check is None else enable_selector_check
        self.enable_mag_check = enable_vision if enable_mag_check is None else enable_mag_check

        self.bridge: BridgeInterface = bridge or BridgeInterface(enabled=False)
        if mission_label:
            self.bridge.context.setdefault("mission_label", mission_label)
        if request_id:
            self.bridge.context.setdefault("request_id", str(request_id))
        if site:
            self.bridge.context.setdefault("site", site)
        self.auto_mode = bool(auto_mode or self.bridge.enabled)
        self.mission_label = mission_label
        self.request_id = request_id
        self.site = site

        if not self.enable_vision:
            self.enable_qr_check = False
            self.enable_selector_check = False
            self.enable_mag_check = False

        self.any_vision_check = (
            self.enable_qr_check or
            self.enable_selector_check or
            self.enable_mag_check
        )

        self.positions = load_positions_by_label(TEACH_FILE)

        # 최신 JSON에는 새 라벨명이 직접 포함되어 있으므로, 실제로 존재하는
        # 라벨만 별칭으로 등록한다. (예: 새 라벨이 JSON에 있으면 그대로 사용)
        effective_aliases: Dict[str, str] = {}
        unresolved_aliases: List[str] = []
        for src, dst in self.LABEL_ALIASES.items():
            if src in self.positions:
                # JSON이 이미 새 라벨을 제공하므로 별칭이 필요 없다.
                continue
            if dst in self.positions:
                effective_aliases[src] = dst
            else:
                unresolved_aliases.append(f"{src}->{dst}")
        if unresolved_aliases:
            print("[WARN] JSON에서 찾을 수 없는 별칭 라벨:", unresolved_aliases)
        self.label_aliases = effective_aliases
        missing = [lbl for lbl in self.REQUIRED_LABELS if lbl not in self.positions]
        if missing:
            print("[WARN] JSON에서 찾지 못한 라벨:", missing)

        self.bc: BridgeClient = None
        self.vision: VisionController = None
        self.gripper_actions: Dict[str, str] = {}

        self._define_blocks()
        self._setup_default_gripper_policy()

        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()
        self.mag_failure_handler: Optional[Callable[[], None]] = None

    # ----------------------------- 브리지 유틸 -----------------------------
    def _stage_key(self, text: Optional[str]) -> str:
        base = (text or "").lower()
        slug = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
        return slug or "stage"

    def _emit_progress(self, stage: str, message: str, **extra) -> None:
        if not self.bridge.enabled:
            return
        payload = {"stage": stage, "message": message}
        if extra:
            payload.update(extra)
        self.bridge.emit("progress", **payload)

    def _emit_log(self, stage: str, message: str, level: str = "info", **extra) -> None:
        if not self.bridge.enabled:
            return
        payload = {"stage": stage, "message": message, "level": level}
        if extra:
            payload.update(extra)
        self.bridge.emit("log", **payload)

    def _emit_complete(self, status: str, message: str, **extra) -> None:
        if not self.bridge.enabled:
            return
        payload = {"status": status, "stage": "completed", "message": message}
        if extra:
            payload.update(extra)
        self.bridge.emit("complete", **payload)

    # ----------------------------- Keep-Alive -----------------------------
    def _send_keepalive(self):
        if not self.bc:
            return
        try:
            if hasattr(self.bc, "keepalive"):
                self.bc.keepalive()
            elif hasattr(self.bc, "ping"):
                self.bc.ping()
            else:
                _ = self.bc.hello_info
        except Exception:
            pass

    def _keepalive_loop(self):
        while not self._keepalive_stop.is_set():
            self._send_keepalive()
            self._keepalive_stop.wait(KEEPALIVE_INTERVAL)

    def start_keepalive(self):
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            return
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()

    def stop_keepalive(self):
        self._keepalive_stop.set()
        if self._keepalive_thread:
            self._keepalive_thread.join(timeout=1.0)

    # ----------------------------- 레일 제어 -----------------------------
    def rail_ensure_extended(self):
        print("[RAIL] 레일 위치 확인 및 배출...")
        current_pos = self.bc.rail_get_position()
        print(f"[RAIL] 현재 위치: {current_pos:.1f}mm")

        if current_pos < RAIL_MAX_STROKE * 0.9:
            print(f"[RAIL] 레일이 내부에 있습니다. 외부로 배출합니다... (절대이동 M2)")
            result = self.bc.rail_move_to(
                position=RAIL_MAX_STROKE,
                speed=RAIL_FORWARD_SPEED,
                wait=True
            )
            if result.get("ok"):
                print(f"[RAIL] ✓ 레일 배출 완료. 위치: {result.get('current_position')}mm")
            else:
                print(f"[RAIL] ✗ 레일 배출 실패: {result.get('err')}")
                raise RuntimeError("레일 배출 실패")
        else:
            print(f"[RAIL] 레일이 이미 외부에 있습니다.")

    def rail_ensure_home(self):
        print("[RAIL] 레일 위치 확인 및 인입...")
        current_pos = self.bc.rail_get_position()
        print(f"[RAIL] 현재 위치: {current_pos:.1f}mm")

        if current_pos > 20.0:
            print(f"[RAIL] 레일이 외부에 있습니다. HOME(0mm)으로 인입합니다...")
            result = self.bc.rail_home(speed=RAIL_HOME_SPEED, wait=True)
            if result.get("ok"):
                print(f"[RAIL] ✓ 레일 인입 완료. 위치: {result.get('current_position')}mm")
            else:
                print(f"[RAIL] ✗ 레일 인입 실패: {result.get('err')}")
                raise RuntimeError("레일 인입 실패")
        else:
            print("[RAIL] 레일이 이미 HOME 위치에 있습니다.")

    def _execute_lockdown_recovery(self, stage: str, reason: str, meta: Optional[dict] = None) -> None:
        info_meta = meta or {}
        self._emit_log(stage, "락다운 복구 절차 실행", level='warning', meta=info_meta)
        print("\n[LOCKDOWN] 자동 복구 절차를 시작합니다...")
        try:
            if self.with_mag:
                try:
                    self.rail_ensure_extended()
                except Exception as err:
                    self._emit_log(stage, f"레일 배출 실패: {err}", level='error', meta=info_meta)
                    print(f"[LOCKDOWN] 레일 배출 중 오류: {err}")
                try:
                    self._recover_mag_failure_outbound()
                    self._emit_log(stage, "탄창을 원위치에 복귀했습니다.", level='info', meta=info_meta)
                except Exception as err:
                    self._emit_log(stage, f"탄창 복귀 실패: {err}", level='error', meta=info_meta)
                    print(f"[LOCKDOWN] 탄창 복귀 중 오류: {err}")
            try:
                self.bc.gripper_open()
            except Exception as err:
                self._emit_log(stage, f"그리퍼 개방 실패: {err}", level='warning', meta=info_meta)
            try:
                self.move_to(self.home_label, desc="lockdown home")
                self._emit_log(stage, "로봇을 home_rdy 자세로 이동했습니다.", level='info', meta=info_meta)
            except Exception as err:
                self._emit_log(stage, f"home_rdy 이동 실패: {err}", level='error', meta=info_meta)
                print(f"[LOCKDOWN] home_rdy 이동 중 오류: {err}")
            try:
                self.rail_ensure_home()
            except Exception as err:
                self._emit_log(stage, f"레일 인입 실패: {err}", level='warning', meta=info_meta)
        finally:
            print("[LOCKDOWN] 복구 절차 종료. 관리자 해제를 기다립니다.")

    def _trigger_lockdown(self, stage: str, message: str, reason: str, meta: Optional[dict] = None) -> None:
        payload = dict(meta or {})
        payload.setdefault("timestamp", datetime.utcnow().isoformat())
        payload.setdefault("mission", self.mission_number)
        payload.setdefault("direction", self.direction)
        payload.setdefault("withMag", self.with_mag)
        if self.bridge.enabled:
            self.bridge.emit(
                "lockdown",
                stage=stage,
                message=message,
                reason=reason,
                meta=payload
            )
        self._execute_lockdown_recovery(stage, reason, payload)

    def rail_retract(self):
        stage_key = self._stage_key("레일 인입")
        attempt = 1
        while True:
            print(f"[RAIL] 레일 인입 시도 #{attempt}...")
            result = self.bc.rail_home(speed=RAIL_HOME_SPEED, wait=True)
            if result.get("ok"):
                current_pos = result.get("current_position")
                print(f"[RAIL] ✓ 레일 인입 완료. 위치: {current_pos}mm")
                if self.bridge.enabled:
                    self._emit_progress(stage_key, "레일 인입 완료", attempt=attempt, position=current_pos)
                return

            err = result.get("err") or "알 수 없는 오류"
            print(f"[RAIL] ✗ 레일 인입 실패: {err}")
            if not (self.auto_mode and self.bridge.enabled):
                raise RuntimeError("레일 인입 실패")

            self._emit_log(stage_key, f"레일 인입 실패: {err}", level="error", attempt=attempt)
            self._emit_progress(stage_key, "레일 인입 실패 - 사용자 확인 대기", attempt=attempt, error=err)

            # 레일을 다시 외부 위치로 이동시켜 사용자가 상태를 점검할 수 있도록 한다.
            try:
                extend_result = self.bc.rail_move_to(
                    position=RAIL_MAX_STROKE,
                    speed=RAIL_FORWARD_SPEED,
                    wait=True
                )
                if extend_result.get("ok"):
                    self._emit_progress(
                        stage_key,
                        "레일을 다시 외부 위치로 이동했습니다.",
                        attempt=attempt,
                        position=extend_result.get("current_position")
                    )
                else:
                    self._emit_log(
                        stage_key,
                        f"레일 외부 재이동 실패: {extend_result.get('err')}",
                        level="warning",
                        attempt=attempt
                    )
            except Exception as extend_exc:
                self._emit_log(
                    stage_key,
                    f"레일 외부 재이동 중 예외: {extend_exc}",
                    level="warning",
                    attempt=attempt
                )

            prompt = (
                "레일 인입에 실패했습니다. 장애물을 확인한 뒤 단말의 '다시 시도' 버튼을 눌러주세요."
            )
            meta = {
                "attempt": attempt,
                "error": err,
                "direction": self.direction,
                "mission": self.mission_number,
                "withMag": self.with_mag,
                "buttonLabel": "다시 시도"
            }
            try:
                self.bridge.wait_for("rail_retract_retry", prompt, allow_cancel=True, meta=meta)
            except BridgeAbort:
                raise

            attempt += 1
            time.sleep(0.5)

    def rail_extend(self):
        print("[RAIL] 레일 배출 시작...")
        result = self.bc.rail_move_to(RAIL_MAX_STROKE, speed=RAIL_FORWARD_SPEED, wait=True)
        if result.get("ok"):
            print(f"[RAIL] ✓ 레일 배출 완료. 위치: {result.get('current_position')}mm")
        else:
            print(f"[RAIL] ✗ 레일 배출 실패: {result.get('err')}")
            raise RuntimeError("레일 배출 실패")
        
    def rail_wait_outside(self, wait_seconds: float = 10.0):
        print("[RAIL] 불입 준비 - 레일을 외부 위치로 이동합니다.")
        try:
            current_pos = self.bc.rail_get_position()
        except Exception:
            current_pos = 0.0

        if current_pos < RAIL_MAX_STROKE - 5.0:
            print(f"[RAIL] 현재 {current_pos:.1f}mm → 외부 {RAIL_MAX_STROKE:.0f}mm로 이동합니다.")
            result = self.bc.rail_move_to(
                position=RAIL_MAX_STROKE,
                speed=RAIL_FORWARD_SPEED,
                wait=True
            )
            if not result.get("ok"):
                raise RuntimeError(f"레일 외부 이동 실패: {result.get('err')}")
            current_pos = result.get("current_position", RAIL_MAX_STROKE)
        else:
            print(f"[RAIL] 레일이 이미 외부({current_pos:.1f}mm) 위치에 있습니다. 유지 대기합니다.")

        stage_key = self._stage_key("rail_outside")
        self._emit_progress(stage_key, "레일 외부 위치 확보", position=current_pos)
        if self.auto_mode:
            prompt = "레일 위에 총기와 탄약을 올리고 준비가 되면 단말에서 확인 버튼을 눌러주세요."
            try:
                self.bridge.wait_for("return_load", prompt, meta={
                    "direction": self.direction,
                    "mission": self.mission_number,
                    "withMag": self.with_mag
                })
            except BridgeAbort:
                raise
        else:
            print(f"[RAIL] 외부 위치에서 {wait_seconds:.1f}초 대기합니다...")
            time.sleep(wait_seconds)
            print("[RAIL] 대기 완료.")
        self._emit_progress(stage_key, "사용자 준비 완료", completed=True)

    # ----------------------------- 비전 검사 래퍼 -----------------------------
    def vision_check_qr(self):
        """QR 코드 검사 - 불일치 시 프로세스 중단 및 반출"""
        if not self.enable_qr_check:
            reason = "비전 검사 비활성화됨" if not self.enable_vision else "사용자 설정"
            print(f"[VISION] {reason} - QR 검사 스킵")
            return
        
        print("\n" + "="*60)
        print("[비전 검사] QR 코드 확인")
        print("="*60)
        
        is_valid, message = self.vision.check_qr_code(self.expected_qr, display=True)
        print(f"[VISION] {message}")
        
        if not is_valid:
            print("\n" + "!"*60)
            print("[경고] QR 코드 검증 실패!")
            print("!"*60)
            print("[조치] 총기를 반출합니다...")
            
            # 반출 프로세스 실행
            self._execute_return_process(cycle_back=True)
            raise RuntimeError(f"QR 코드 검증 실패: {message}")
        
        print("[VISION] ✓ QR 코드 검증 통과\n")

    def vision_check_selector(self):
        """조정간 안전 상태 검사 - SAFE 아니면 프로세스 중단 및 반출"""
        if not self.enable_selector_check:
            reason = "비전 검사 비활성화됨" if not self.enable_vision else "사용자 설정"
            print(f"[VISION] {reason} - 조정간 검사 스킵")
            return
        
        print("\n" + "="*60)
        print("[비전 검사] 조정간 안전 상태 확인")
        print("="*60)
        stage_key = self._stage_key("selector_check")

        if self.auto_mode:
            attempt = 0
            while True:
                attempt += 1
                is_safe, message = self.vision.check_selector_safe(display=True)
                print(f"[VISION] {message}")
                self._emit_progress(stage_key, message, attempt=attempt)
                if is_safe:
                    success_msg = "조정간 SAFE 확인 완료"
                    print(f"[VISION] ✓ {success_msg}\n")
                    self._emit_progress(stage_key, success_msg, attempt=attempt, completed=True)
                    return

                warn_msg = "조정간이 안전 상태가 아닙니다!"
                print("\n" + "!"*60)
                print(f"[경고] {warn_msg}")
                print("!"*60)
                self._emit_log(stage_key, message, level='warning', attempt=attempt)

                try:
                    self.rail_extend()
                except Exception as err:
                    self._emit_log(stage_key, f"레일 배출 실패: {err}", level='error', attempt=attempt)
                    raise

                try:
                    self.bridge.wait_for(
                        "selector_retry",
                        "조정간을 SAFE 위치로 맞춘 뒤 단말의 버튼을 눌러주세요.",
                        allow_cancel=True,
                        meta={"attempt": attempt}
                    )
                except BridgeAbort:
                    raise

                try:
                    self.rail_retract()
                except Exception as err:
                    self._emit_log(stage_key, f"레일 인입 실패: {err}", level='error', attempt=attempt)
                    raise
        else:
            is_safe, message = self.vision.check_selector_safe(display=True)
            print(f"[VISION] {message}")

            if not is_safe:
                print("\n" + "!"*60)
                print("[경고] 조정간이 안전 상태가 아닙니다!")
                print("!"*60)
                print("[조치] 총기를 반출합니다...")

                # 반출 프로세스 실행
                self._execute_return_process(cycle_back=True)
                raise RuntimeError(f"조정간 안전 검사 실패: {message}")

            print("[VISION] ✓ 조정간 안전 상태 확인 완료\n")

    def vision_check_mag(self):
        """탄창 방향 검사 - 우상탄 아니면 프로세스 정지"""
        if not self.enable_mag_check:
            reason = "비전 검사 비활성화됨" if not self.enable_vision else "사용자 설정"
            print(f"[VISION] {reason} - 탄창 검사 스킵")
            return
        
        print("\n" + "="*60)
        print("[비전 검사] 탄창 방향 확인")
        print("="*60)
        
        is_correct, message = self.vision.check_mag_orientation(display=True)
        print(f"[VISION] {message}")
        stage_key = self._stage_key("mag_check")
        self._emit_progress(stage_key, message)

        if not is_correct:
            print("\n" + "!"*60)
            print("[경고] 탄창 방향이 올바르지 않습니다!")
            print("!"*60)
            print("[조치] 탄창을 원위치에 복귀하고 로봇을 정지합니다.")

            if "좌상탄" in message:
                self._trigger_lockdown(
                    stage="mag_check",
                    message=message,
                    reason="left_round_detected",
                    meta={
                        "direction": self.direction,
                        "mission": self.mission_number,
                        "withMag": self.with_mag
                    }
                )
                raise BridgeAbort("좌상탄 감지")

            if self.mag_failure_handler:
                try:
                    print("[조치] 탄창을 원래 위치로 복귀합니다...")
                    self.mag_failure_handler()
                    print("[조치] 탄창 복귀 완료. 사용자 확인이 필요합니다.")
                except Exception as e:
                    print(f"[경고] 탄창 복귀 중 오류: {e}")
            else:
                print("[경고] 복귀 시퀀스가 설정되어 있지 않습니다. 수동으로 복구하세요.")
            raise RuntimeError(f"탄창 방향 검사 실패: {message}")

        success_msg = "탄창 방향 확인 완료 (우상탄)"
        print(f"[VISION] ✓ {success_msg}\n")
        self._emit_progress(stage_key, success_msg, completed=True)

    def _execute_return_process(self, cycle_back: bool = True):
        """검사 실패 시 총기 반출 프로세스"""
        print("\n[반출 프로세스] 시작...")
        try:
            # 현재 위치에서 안전하게 총기를 레일로 복귀
            # 실제 구현은 현재 상태에 따라 달라질 수 있음
            print("[반출] 레일 배출...")
            self.rail_extend()
            if cycle_back:
                print("[반출] 레일을 다시 HOME으로 인입합니다...")
                self.rail_retract()
            print("[반출] 완료. 총기를 확인하고 재시도하세요.")
        except Exception as e:
            print(f"[반출 오류] 반출 중 오류 발생: {e}")

    # ----------------------------- 블록 정의 -----------------------------
    def _define_blocks(self):
        # rifle1
        self.block_pick_rifle1 = ["pick_rifle1", "grip_rifle1", "up_rifle1"]
        
        # rifle2
        self.block_pick_rifle2 = ["pick_rifle2", "grip_rifle2", "up1_rifle2", 
                                 "up2_rifle2", "up3_rifle2", "up4_rifle2"]
        
        # 레일 위치
        self.block_up_before_rail = ["up1_rifle", "up2_rifle"]
        self.block_place_on_rail = ["up1_rifle", "up2_rifle",
                                   "on_rail1", "on_rail2", "on_rail3",
                                   "on_rail4", "on_rail5", "on_rail6", "fallback_rail"]
        
        # 조정간 - 단발
        self.block_selector_approach = ["go_selector1", "go_selector2", "go_selector3"]
        self.block_selector_spin_semi = ["spin_selector_semi"]
        self.block_selector_out = ["out_selector1", "out_selector2"]
        self.block_selector_single = (self.block_selector_approach + 
                                     self.block_selector_spin_semi + 
                                     self.block_selector_out)
        
        # 조정간 - 안전 (격발 후)
        self.block_selector_from_fire = ["go_selector_from_fire1", "go_selector_from_fire2"]
        self.block_selector_spin_safe = ["spin_selector_safe"]
        self.block_selector_safe_out = ["out_selector_safe_end1", "out_selector_safe_end2"]
        self.block_selector_safe = (self.block_selector_from_fire + 
                                   self.block_selector_spin_safe + 
                                   self.block_selector_safe_out)
        
        # 장전 & 격발
        self.block_cocking_approach = ["go_cocking1", "go_cocking2", "go_cocking3"]
        self.block_cocking_action = ["cocking"]
        self.block_cocking_out = ["go_cocking3", "out_cocking1", "out_cocking2"]
        self.block_cocking_core = (self.block_cocking_approach + 
                                  self.block_cocking_action + 
                                  self.block_cocking_out)
        
        self.block_fire_approach = ["go_fire1", "go_fire2"]
        self.block_fire_action = ["fire"]
        self.block_fire_out = ["out_fire1", "out_fire2"]
        self.block_fire_core = self.block_fire_approach + self.block_fire_action + self.block_fire_out
        
        # 탄창
        self.block_pick_mag1 = ["go2mag1", "grip_mag1", "pickup_mag1"]
        self.block_pick_mag2 = ["go_mag2", "grip_mag2", "pickup_mag2", "out_mag2"]
        self.block_place_mag_on_rail = [
            "go_mag_rail1", "go_mag_rail2", "go_mag_rail3_vision",
            "go_mag_rail4", "go_mag_rail5", "go_mag_rail6",
            "pickdown_rail", "out_mag_rail1", "out_mag_rail2", "out_mag_rail3"
        ]
        self.block_retrieve_mag_from_rail = ["pickdown_rail", "out_mag_rail1", 
                                            "out_mag_rail2", "out_mag_rail3"]
        
        # 불입 전용 - 탄창 비전 검사 후 배치
        self.block_mag_to_vision = ["go_pick_mag_return", "o_mag2vision_return1", 
                                   "go_mag2vision_return2", "check_vision_mag_return"]
        self.block_vision_to_mag = ["vision2mag_return"]
        self.block_place_mag_return = ["place_mag1_return1", "place_mag1_return2"]
        
        # 불입 전용 - 총기 반환
        self.block_return1_rifle_pickup = ["up1_rifle", "up2_rifle"]
        self.block_return1_rifle_place = self.block_pick_rifle1[::-1]
        self.block_return2_rifle_pickup = ["up1_rifle", "up2_rifle"]
        self.block_return2_rifle_place = self.block_pick_rifle2[::-1]
        
        self.home_label = "home_rdy"

    def _setup_default_gripper_policy(self):
        # rifle1
        self.gripper_actions["grip_rifle1"] = "close"
        
        # rifle2
        self.gripper_actions["grip_rifle2"] = "close"
        
        # 탄창
        self.gripper_actions["grip_mag1"] = "close"
        self.gripper_actions["grip_mag2"] = "close"
        self.gripper_actions["pickdown_rail"] = "close"
        
        # 레일 위치
        self.gripper_actions["fallback_rail"] = "open"

    # ----------------------------- 로봇 제어 -----------------------------
    def connect_bridge(self):
        print(f"[연결] BridgeClient 연결 중... host={self.bridge_host}")
        # 긴 동작 구간에서 소켓 타임아웃이 끊기지 않도록 IO 타임아웃을 넉넉하게 확장한다.
        self.bc = BridgeClient(host=self.bridge_host, io_timeout=30.0, retry=1)
        try:
            hello = self.bc.connect()
            # hello 예: {"ok":true,"hello":"robot_bridge","ver":"1.3", ...}
            if isinstance(hello, dict):
                ver = hello.get("ver") or hello.get("version") or "?"
                name = hello.get("hello") or "robot_bridge"
                print(f"[연결] BridgeClient 연결 완료. {name} ver={ver}")
            else:
                print(f"[연결] BridgeClient 연결 완료.")
            try:
                # 아두이노 측 버퍼를 비워 레일 명령이 즉시 수행되도록 한다.
                self.bc.arduino_flush()
            except Exception:
                pass
        except Exception as e:
            # 연결 실패 시 명확한 원인 안내
            self.bc = None
            raise RuntimeError(f"브릿지({self.bridge_host}) 연결 실패: {e}")

    def resolve_label(self, label: str) -> str:
        if label in self.positions:
            return label
        target = self.label_aliases.get(label)
        if target and target in self.positions:
            return target
        return label

    def move_to(self, label: str, desc: str = ""):
        target_label = self.resolve_label(label)
        if target_label not in self.positions:
            raise ValueError(f"라벨 '{label}'(실제 '{target_label}')을 positions에서 찾을 수 없습니다.")

        pos_data = self.positions[target_label]
        j6 = [
            pos_data.get("Joint1(deg)", 0.0),
            pos_data.get("Joint2(deg)", 0.0),
            pos_data.get("Joint3(deg)", 0.0),
            pos_data.get("Joint4(deg)", 0.0),
            pos_data.get("Joint5(deg)", 0.0),
            pos_data.get("Joint6(deg)", 0.0)
        ]
        
        info_str = f"move_to('{label}')"
        if desc:
            info_str += f" - {desc}"
        if target_label != label:
            info_str += f" [alias→'{target_label}']"
        print(f"[이동] {info_str}")
        
        self.bc.movej(j6, blocking=True)
        
        action = self.gripper_actions.get(label)
        if action is None and target_label != label:
            action = self.gripper_actions.get(target_label)
        if action == "open":
            print(f"  → gripper OPEN at '{label}'")
            self.bc.gripper_open()
        elif action == "close":
            print(f"  → gripper CLOSE at '{label}'")
            self.bc.gripper_close()

    def exec_labels(self, labels: List[str], block_name: str = ""):
        info = f"블록 '{block_name}'" if block_name else "라벨 시퀀스"
        print(f"\n[실행] {info} 시작 -> {len(labels)}개 라벨")
        for lbl in labels:
            self.move_to(lbl, desc=f"in block '{block_name}'")
        print(f"[실행] {info} 완료.\n")

    def _manual_gripper(self, action: Optional[str], label: str):
        if action == "open":
            print(f"  → gripper OPEN (sequence) at '{label}'")
            self.bc.gripper_open()
        elif action == "close":
            print(f"  → gripper CLOSE (sequence) at '{label}'")
            self.bc.gripper_close()

    def _run_sequence(self, seq: List[Any], desc: str):
        for entry in seq:
            if isinstance(entry, tuple):
                label, action = entry
            else:
                label, action = entry, None
            if label == "__VISION_MAG__":
                self.vision_check_mag()
                continue
            self.move_to(label, desc=desc)
            if action:
                self._manual_gripper(action, label)

    @staticmethod
    def _sequence_labels(seq: List[Any]) -> List[str]:
        labels: List[str] = []
        for entry in seq:
            label = entry[0] if isinstance(entry, tuple) else entry
            if label == "__VISION_MAG__":
                continue
            labels.append(label)
        return labels

    @contextmanager
    def _mag_failure_guard(self, handler: Callable[[], None]):
        prev = self.mag_failure_handler
        self.mag_failure_handler = handler
        try:
            yield
        finally:
            self.mag_failure_handler = prev

    @contextmanager
    def _suspend_gripper_actions(self, labels: Iterable[str]):
        saved: Dict[str, str] = {}
        for lbl in labels:
            if lbl in self.gripper_actions:
                saved[lbl] = self.gripper_actions.pop(lbl)
        try:
            yield
        finally:
            for lbl, action in saved.items():
                self.gripper_actions[lbl] = action

    # ----------------------------- 조정간/장전/격발 블록 -----------------------------
    def run_selector_single(self):
        block_name = "selector->single"
        labels = self.block_selector_single
        print(f"\n[실행] 블록 '{block_name}' 시작 -> {len(labels)}개 라벨")
        self.move_to("go_selector1", desc=f"in block '{block_name}'")
        self.move_to("go_selector2", desc=f"in block '{block_name}'")
        print("  → gripper CLOSE (manual) before 'go_selector3'")
        self.bc.gripper_close()
        self.move_to("go_selector3", desc=f"in block '{block_name}'")
        self.move_to("spin_selector_semi", desc=f"in block '{block_name}'")
        self.move_to("out_selector1", desc=f"in block '{block_name}'")
        print("  → gripper OPEN (manual) after 'out_selector1'")
        self.bc.gripper_open()
        self.move_to("out_selector2", desc=f"in block '{block_name}'")
        print(f"[실행] 블록 '{block_name}' 완료.\n")

    def run_selector_safe_from_fire(self):
        block_name = "selector->safe(from fire)"
        labels = self.block_selector_safe
        print(f"\n[실행] 블록 '{block_name}' 시작 -> {len(labels)}개 라벨")
        self.move_to("go_selector_from_fire1", desc=f"in block '{block_name}'")
        print("  → gripper CLOSE (manual) before 'go_selector_from_fire2'")
        self.bc.gripper_close()
        self.move_to("go_selector_from_fire2", desc=f"in block '{block_name}'")
        self.move_to("spin_selector_safe", desc=f"in block '{block_name}'")
        self.move_to("out_selector_safe_end1", desc=f"in block '{block_name}'")
        self.move_to("out_selector_safe_end2", desc=f"in block '{block_name}'")
        print("  → gripper OPEN (manual) after 'out_selector_safe_end2'")
        self.bc.gripper_open()
        print(f"[실행] 블록 '{block_name}' 완료.\n")

    def run_cocking(self):
        self.exec_labels(self.block_cocking_core, "cocking")

    def run_fire(self):
        self.exec_labels(self.block_fire_core, "fire")

    # ----------------------------- 불입 프로세스 시퀀스 정의 -----------------------------
    def _get_return_mag_pickup_sequence(self) -> List[Any]:
        return [
            "go_pick_mag_return",
            "grip_mag_return",
            ("pickup_mag_return", "close"),
            "go_mag2vision1_return",
            "go_mag2vision2_return",
            "check_vision_mag_return",
            "__VISION_MAG__",
            "go_vision2mag_return",
        ]

    def _get_return_mag_place_sequence(self) -> List[Any]:
        if self.mission_number == 1:
            return [
                "place_mag1_1_return",
                ("place_mag1_2_return", "open"),
                "out_mag1_1_return",
                "out_mag1_2_return",
            ]
        return [
            "out_mag2",
            "pickup_mag2",
            ("grip_mag2", "open"),
            "go_mag2",
            "out_mag2",
        ]
    def _recover_mag_failure_inbound(self):
        print("[복구] 탄창을 반납 위치로 되돌립니다...")
        seq: List[Any] = [
            "go_mag2vision2_return",
            "go_mag2vision1_return",
            ("pickup_mag_return", "open"),
            "grip_mag_return",
            "go_pick_mag_return1",
        ]
        self._run_sequence(seq, "mag failure recovery (return)")
        self.bc.gripper_open()

    def _recover_mag_failure_outbound(self):
        print("[복구] 탄창을 보관 위치로 되돌립니다...")
        if self.mission_number == 1:
            seq: List[Any] = [
                "go_mag_rail2",
                "go_mag_rail1",
                ("pickup_mag1", "open"),
                "grip_mag1",
                "go2mag1",
            ]
            suspend = ["grip_mag1"]
        else:
            seq = [
                "go_mag_rail2",
                "go_mag_rail1",
                "out_mag2",
                ("pickup_mag2", "open"),
                "grip_mag2",
                "go_mag2",
            ]
            suspend = ["grip_mag2"]

        with self._suspend_gripper_actions(suspend):
            self._run_sequence(seq, "mag failure recovery (out)")
        self.bc.gripper_open()

    def _get_return_rifle1_pickup_sequence(self) -> List[Any]:
        return [
            "go_selector1",
            ("go_selector2", "close"),
            "go_selector3",
            "spin_selector_semi",
            "out_selector1",
            "out_selector2",
            ("go_cocking1", "open"),
            "go_cocking2",
            "go_cocking3",
            "cocking",
            "go_cocking3",
            "cocking",
            "go_cocking3",
            "cocking",
            "go_cocking3",
            "out_cocking1",
            "out_cocking2",
            "go_fire1",
            "go_fire2",
            "fire",
            "out_fire1",
            "out_fire2",
            "go_cocking2",
            "go_cocking3",
            "cocking",
            "go_cocking3",
            "out_cocking1",
            "out_cocking2",
            "up2_rifle",
            "out_fire2selector",
            ("go_selector_from_fire1", "close"),
            "go_selector_from_fire2",
            "spin_selector_safe",
            ("out_selector_safe_end1", "open"),
            "out_selector_safe_end2",
            "rifle_pickgo1_return",
            "rifle_pickgo2_return",
            ("rifle_pickgo3_return", "close"),
            "rifle_pickgo4_return",
        ]

    def _get_return_rifle1_place_sequence(self) -> List[Any]:
        return [
            "rifle2_pickgo1_return",
            "rifle1_pickgo1_return",
            "rifle1_pickgo2_return",
            "rifle1_pickgo3_return",
            "rifle1_pickgo4_return",
            "rifle1_pickgo5_return",
            ("rifle1_pickgo6_return", "open"),
            "rifle1_pickgo7_return",
        ]

    def _get_return_rifle2_pickup_sequence(self) -> List[Any]:
        seq: List[Any] = [
            "go_selector1",
            ("go_selector2", "close"),
            "go_selector3",
            "spin_selector_semi",
            "out_selector1",
            "out_selector2",
            ("go_cocking1", "open"),
            "go_cocking2",
            "go_cocking3",
            "cocking",
            "go_cocking3",
            "cocking",
            "go_cocking3",
            "cocking",
            "go_cocking3",
            "out_cocking1",
            "out_cocking2",
            "go_fire1",
            "go_fire2",
            "fire",
            "out_fire1",
            "out_fire2",
            "go_cocking2",
            "go_cocking3",
            "cocking",
            "go_cocking3",
            "out_cocking1",
            "out_cocking2",
            "up2_rifle",
            "out_fire2selector",
            ("go_selector_from_fire1", "close"),
            "go_selector_from_fire2",
            "spin_selector_safe",
            ("out_selector_safe_end1", "open"),
            "out_selector_safe_end2",
            "rifle_pickgo1_return",
            "rifle_pickgo2_return",
            ("rifle_pickgo3_return", "close"),
            "rifle_pickgo4_return",
            "rifle2_pickgo1_return",
            "rifle1_pickgo1_return",
            "rifle2_pickgo3_return",
        ]

        if self.with_mag:
            return ["go_mag2rifle_return"] + seq
        return seq

    def _get_return_rifle2_place_sequence(self) -> List[Any]:
        return [
            "rifle2_pickgo4_return",
            "rifle2_pickgo5_return",
            "up2_rifle2",
            "up1_rifle2",
            ("grip_rifle2", "open"),
            "pick_rifle2",
        ]

    # ----------------------------- 불입 프로세스 블록 -----------------------------
    def return_mag_pickup_and_check(self):
        """탄창 집기 후 비전으로 이동 -> 우상탄 확인"""
        print("\n[불입-탄창] 탄창 집기 및 비전 위치 이동")
        seq = self._get_return_mag_pickup_sequence()
        with self._mag_failure_guard(self._recover_mag_failure_inbound):
            self._run_sequence(seq, "return mag pickup/vision")

    def return_mag_place(self):
        """탄창을 보관함에 내려놓기"""
        print("\n[불입-탄창] 탄창 반납 배치")
        seq = self._get_return_mag_place_sequence()
        self._run_sequence(seq, "return mag place")

    def return_rifle_1_pickup(self):
        """총기1 레일에서 집기"""
        print("\n[불입-총기1] 레일에서 집기")
        seq = self._get_return_rifle1_pickup_sequence()
        self._run_sequence(seq, "return rifle1 pickup")

    def return_rifle_1_place(self):
        """총기1 보관함에 반납"""
        print("\n[불입-총기1] 보관함 반납")
        seq = self._get_return_rifle1_place_sequence()
        self._run_sequence(seq, "return rifle1 place")

    def return_rifle_2_pickup(self):
        """총기2 레일에서 집기"""
        print("\n[불입-총기2] 레일에서 집기")
        seq = self._get_return_rifle2_pickup_sequence()
        self._run_sequence(seq, "return rifle2 pickup")

    def return_rifle_2_place(self):
        """총기2 보관함에 반납"""
        print("\n[불입-총기2] 보관함 반납")
        seq = self._get_return_rifle2_place_sequence()
        self._run_sequence(seq, "return rifle2 place")

    # ----------------------------- 스텝 빌드 -----------------------------
    def _run_steps(self, steps: List[Step]):
        total = len(steps)
        for idx, step in enumerate(steps, start=1):
            print("\n" + "="*60)
            print(f"[단계 {idx}/{total}] {step.title}")
            print("="*60)
            print(f"설명: {step.description}")
            
            if step.preview_labels:
                print(f"관련 라벨: {', '.join(step.preview_labels)}")
            
            user_input = input("\n실행하시겠습니까? (y=실행, s=스킵, q=종료): ").strip().lower()
            
            if user_input == 'q':
                print("[중단] 사용자가 종료를 선택했습니다.")
                break
            elif user_input == 's':
                print(f"[스킵] 단계 {idx} 스킵됨.")
                continue
            elif user_input == 'y':
                print(f"[실행] 단계 {idx} 실행 중...")
                try:
                    step.run_fn()
                    print(f"[완료] 단계 {idx} 완료.")
                except Exception as e:
                    print(f"[오류] 단계 {idx} 실행 중 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    retry = input("다시 시도하시겠습니까? (y/n): ").strip().lower()
                    if retry == 'y':
                        try:
                            step.run_fn()
                            print(f"[완료] 단계 {idx} 재시도 완료.")
                        except Exception as e2:
                            print(f"[오류] 재시도 실패: {e2}")
                            raise
                    else:
                        raise
            else:
                print("[경고] 잘못된 입력입니다. 단계를 스킵합니다.")

    def _run_steps_auto(self, steps: List[Step]):
        total = len(steps)
        for idx, step in enumerate(steps, start=1):
            stage_key = self._stage_key(step.title or f"step-{idx}")
            print("\n" + "="*60)
            print(f"[단계 {idx}/{total}] {step.title}")
            print("="*60)
            print(f"설명: {step.description}")
            if step.preview_labels:
                print(f"관련 라벨: {', '.join(step.preview_labels)}")

            self._emit_progress(stage_key, step.description or step.title or f"단계 {idx}", index=idx, total=total, title=step.title)
            try:
                step.run_fn()
                print(f"[완료] 단계 {idx} 완료.")
                self._emit_progress(stage_key, f"{step.title} 완료", index=idx, total=total, title=step.title, completed=True)
            except BridgeAbort:
                raise
            except Exception as e:
                err_msg = f"단계 {idx} 실행 중 오류: {e}"
                print(f"[오류] {err_msg}")
                self._emit_log(stage_key, err_msg, level='error', index=idx, total=total)
                raise

    def build_steps(self) -> List[Step]:
        steps = []
        
        if self.direction == "in":
            # ===== 불입(IN) 프로세스 =====

            # 0) 레일 외부 확보 및 대기
            steps.append(Step(
                "레일 외부 위치 확보",
                "레일을 외부 위치(800mm)로 이동시킨 뒤 10초간 대기합니다.",
                run_fn=self.rail_wait_outside,
                preview_labels=[]
            ))

            # 1) 레일 인입
            steps.append(Step(
                "레일 인입",
                "외부 대기 후 레일을 HOME(0mm)으로 인입합니다.",
                run_fn=self.rail_retract,
                preview_labels=[]
            ))

            # 2) QR 코드 확인
            if self.enable_qr_check:
                steps.append(Step(
                    "QR 코드 확인",
                    f"총기의 QR 코드를 확인합니다. (기대값: {self.expected_qr})",
                    run_fn=self.vision_check_qr,
                    preview_labels=[]
                ))

            # 3) 조정간 안전 상태 확인
            if self.enable_selector_check:
                steps.append(Step(
                    "조정간 안전 상태 확인",
                    "조정간이 SAFE 상태인지 확인합니다.",
                    run_fn=self.vision_check_selector,
                    preview_labels=[]
                ))
            
            # 탄창 포함이면: 탄창 집기 -> 비전 확인 -> 배치
            if self.with_mag:
                steps.append(Step(
                    "탄창 집기 및 방향 확인",
                    "레일에서 탄창을 집어 비전으로 이동 후 우상탄 여부를 확인합니다.",
                    run_fn=self.return_mag_pickup_and_check,
                    preview_labels=self._sequence_labels(self._get_return_mag_pickup_sequence())
                ))
                steps.append(Step(
                    "탄창 반납",
                    "확인된 탄창을 보관함에 반납합니다.",
                    run_fn=self.return_mag_place,
                    preview_labels=self._sequence_labels(self._get_return_mag_place_sequence())
                ))
            
            # 총기 집기 및 반납
            if self.mission_number == 1:
                steps.append(Step(
                    "총기1 집기",
                    "레일에서 총기1을 집습니다.",
                    run_fn=self.return_rifle_1_pickup,
                    preview_labels=self._sequence_labels(self._get_return_rifle1_pickup_sequence())
                ))
                steps.append(Step(
                    "총기1 반납",
                    "총기1을 보관함에 반납합니다.",
                    run_fn=self.return_rifle_1_place,
                    preview_labels=self._sequence_labels(self._get_return_rifle1_place_sequence())
                ))
            else:
                steps.append(Step(
                    "총기2 집기",
                    "레일에서 총기2를 집습니다.",
                    run_fn=self.return_rifle_2_pickup,
                    preview_labels=self._sequence_labels(self._get_return_rifle2_pickup_sequence())
                ))
                steps.append(Step(
                    "총기2 반납",
                    "총기2를 보관함에 반납합니다.",
                    run_fn=self.return_rifle_2_place,
                    preview_labels=self._sequence_labels(self._get_return_rifle2_place_sequence())
                ))

        else:
            # ===== 불출(OUT) 프로세스 =====
            
            # 1) 레일 준비
            steps.append(Step(
                "레일 준비 (인입)",
                "레일이 외부에 있으면 HOME(0mm)으로 인입합니다.",
                run_fn=self.rail_ensure_home,
                preview_labels=[]
            ))
            
            # 총기 집기
            if self.mission_number == 1:
                steps.append(Step(
                    "총기1 집기",
                    "총기1을 보관함에서 집어 올립니다.",
                    run_fn=lambda: self.exec_labels(self.block_pick_rifle1, "pick rifle1"),
                    preview_labels=self.block_pick_rifle1
                ))
            else:
                steps.append(Step(
                    "총기2 집기",
                    "총기2를 보관함에서 집어 올립니다.",
                    run_fn=lambda: self.exec_labels(self.block_pick_rifle2, "pick rifle2"),
                    preview_labels=self.block_pick_rifle2
                ))

            # 레일에 올리기
            if self.with_mag:
                def place_rifle_on_rail_with_open_after_onrail6():
                    prev = self.gripper_actions.get("on_rail6", None)
                    self.gripper_actions["on_rail6"] = "open"
                    try:
                        self.exec_labels(
                            self.block_place_on_rail,
                            "place rifle on rail (open at on_rail6, then fallback_rail)"
                        )
                    finally:
                        if prev is None:
                            self.gripper_actions.pop("on_rail6", None)
                        else:
                            self.gripper_actions["on_rail6"] = prev

                steps.append(Step(
                    "레일에 총기 올리기",
                    "총기를 레일 상의 지정 위치로 옮겨 내려놓습니다. (on_rail6 후 gripper OPEN)",
                    run_fn=place_rifle_on_rail_with_open_after_onrail6,
                    preview_labels=self.block_place_on_rail
                ))
            else:
                steps.append(Step(
                    "레일에 총기 올리기",
                    "총기를 레일 상의 지정 위치로 옮겨 내려놓습니다.",
                    run_fn=lambda: self.exec_labels(self.block_place_on_rail, "place rifle on rail"),
                    preview_labels=self.block_place_on_rail
                ))

            # 조정간 단발 전환
            steps.append(Step(
                "조정간 단발 전환",
                "조정간을 단발(격발) 상태로 전환합니다.",
                run_fn=self.run_selector_single,
                preview_labels=self.block_selector_single
            ))

            # 드라이건 후 안전
            def drygun_then_safe():
                for name in self.block_cocking_core:
                    self.move_to(name, desc="drygun: cocking set #1")
                for name in self.block_fire_core:
                    self.move_to(name, desc="drygun: fire cycle")
                second_cocking = ["go_cocking2","go_cocking3","cocking",
                                 "go_cocking3","out_cocking1","out_cocking2"]
                for name in second_cocking:
                    self.move_to(name, desc="drygun: cocking set #2")
                self.move_to("up2_rifle", desc="approach before selector-safe")
                self.move_to("out_fire2selector", desc="back to out_fire2selector")
                self.run_selector_safe_from_fire()

            steps.append(Step(
                "드라이건 테스트 후 안전 상태",
                "장전(7단계) → 격발 → 장전(6단계) → up2_rifle → out_fire2selector → 안전",
                run_fn=drygun_then_safe,
                preview_labels=self.block_cocking_core + self.block_fire_core
                               + ["go_cocking2","go_cocking3","cocking",
                                  "go_cocking3","out_cocking1","out_cocking2"]
                               + ["up2_rifle","out_fire2selector"] + self.block_selector_safe
            ))

            # 탄창 포함이면: 탄창 집기 → 레일 배치
            if self.with_mag:
                if self.mission_number == 1:
                    steps.append(Step(
                        "탄창1 집기",
                        "보관함에서 탄창1을 집습니다.",
                        run_fn=lambda: self.exec_labels(self.block_pick_mag1, "pick mag1"),
                        preview_labels=self.block_pick_mag1
                    ))
                else:
                    steps.append(Step(
                        "탄창2 집기",
                        "보관함에서 탄창2를 집습니다.",
                        run_fn=lambda: self.exec_labels(self.block_pick_mag2, "pick mag2"),
                        preview_labels=self.block_pick_mag2
                    ))
                mag_place_seq = [
                    "go_mag_rail1",
                    "go_mag_rail2",
                    "go_mag_rail3_vision",
                    "__VISION_MAG__",
                    "go_mag_rail4",
                    "go_mag_rail5",
                    "go_mag_rail6",
                    "pickdown_rail",
                    "out_mag_rail1",
                    "out_mag_rail2",
                    "out_mag_rail3",
                ]

                def place_mag_on_rail_with_drop():
                    prev = self.gripper_actions.get("pickdown_rail", None)
                    self.gripper_actions["pickdown_rail"] = "open"
                    try:
                        with self._mag_failure_guard(self._recover_mag_failure_outbound):
                            self._run_sequence(mag_place_seq, "place mag on rail")
                    finally:
                        if prev is None:
                            self.gripper_actions.pop("pickdown_rail", None)
                        else:
                            self.gripper_actions["pickdown_rail"] = prev

                steps.append(Step(
                    "탄창 레일에 올리기",
                    "집은 탄창을 레일 위 지정 위치에 내려놓습니다.",
                    run_fn=place_mag_on_rail_with_drop,
                    preview_labels=self._sequence_labels(mag_place_seq)
                ))

        # 공통: 종료 홈
        steps.append(Step(
            "종료 홈",
            "미션 종료. teach 홈(home_rdy)로 이동합니다.",
            run_fn=lambda: self.move_to(self.home_label, desc="end home"),
            preview_labels=[self.home_label]
        ))
        
        # 불출일 경우 마지막에 레일 배출
        if self.direction == "out":
            steps.append(Step(
                "레일 최종 배출",
                "모든 작업이 완료되었습니다. 레일을 외부로 배출합니다.",
                run_fn=self.rail_extend,
                preview_labels=[]
            ))
        
        return steps

    # ----------------------------- 메인 실행 -----------------------------
    def run(self):
        stage_meta = {
            "direction": self.direction,
            "mission": self.mission_number,
            "withMag": self.with_mag
        }
        try:
            self.bridge.start()
            init_stage = self._stage_key("initialize")
            self._emit_progress(init_stage, "장비 연결 준비", meta=stage_meta)

            # 로봇 연결
            self.connect_bridge()
            self.start_keepalive()
            self.bc.gripper_open()
            self.bc.go_home()
            
            # 비전 초기화
            if self.any_vision_check:
                self._emit_progress(self._stage_key("vision_setup"), "비전 시스템 초기화", meta=stage_meta)
                self.vision = VisionController()
                self.vision.initialize()
            else:
                self.vision = None

            steps = self.build_steps()
            print("\n[GUIDE] 단계별 실행이 시작됩니다. 각 단계는 실행 전 미리보기를 제공합니다.")
            if self.auto_mode:
                self._run_steps_auto(steps)
            else:
                self._run_steps(steps)
            self._emit_complete('success', "미션이 완료되었습니다.", meta=stage_meta)

        except BridgeAbort as abort_exc:
            message = str(abort_exc) or "사용자 중단"
            self._emit_log(self._stage_key("aborted"), message, level='error', meta=stage_meta)
            self._emit_complete('error', message, meta=stage_meta)
            print(f"\n[중단] {message}")
            raise
        except KeyboardInterrupt:
            message = "사용자가 프로그램을 중단했습니다."
            self._emit_log(self._stage_key("keyboard"), message, level='error', meta=stage_meta)
            self._emit_complete('error', message, meta=stage_meta)
            print(f"\n[중단] {message}")
        except Exception as e:
            err_msg = f"미션 실행 중 오류: {e}"
            self._emit_log(self._stage_key("error"), err_msg, level='error', meta=stage_meta)
            self._emit_complete('error', err_msg, meta=stage_meta)
            print(f"[ERROR] {err_msg}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.stop_keepalive()
            if self.vision:
                self.vision.cleanup()
            if self.bc:
                try:
                    self.bc.close()
                except Exception:
                    pass
            self.bridge.stop()
            print("[완료] 미션 컨트롤러 종료.")


# ----------------------------- 메인 엔트리 -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="비전 통합 미션 컨트롤러")
    parser.add_argument("--mission", type=int, help="미션 번호 (1 또는 2)")
    parser.add_argument("--mission-label", help="미션/보관함 레이블")
    parser.add_argument("--direction", choices=["in", "out"], help="불입(in) 또는 불출(out)")
    parser.add_argument("--with-mag", action="store_true", help="탄창 포함 여부")
    parser.add_argument("--expected-qr", help="QR 기대값")
    parser.add_argument("--bridge-host", default="192.168.1.23", help="브릿지 호스트 주소")
    parser.add_argument("--bridge-mode", action="store_true", help="브릿지 자동 모드 활성화")
    parser.add_argument("--auto", action="store_true", help="자동 실행 모드")
    parser.add_argument("--request-id", help="요청 ID")
    parser.add_argument("--site", help="사이트 식별자")
    parser.add_argument("--interactive", action="store_true", help="강제로 인터랙티브 모드 사용")

    args = parser.parse_args()

    auto_requested = args.bridge_mode or args.auto or args.mission is not None or args.direction or args.with_mag or args.expected_qr or args.request_id

    if auto_requested and not args.interactive:
        mission_num = resolve_mission_number(args.mission, args.mission_label)
        direction = args.direction or "out"
        with_mag = bool(args.with_mag)
        expected_qr = args.expected_qr or (args.request_id if direction == "in" and args.request_id else str(mission_num))
        bridge = BridgeInterface(enabled=args.bridge_mode, context={
            "request_id": args.request_id,
            "mission_label": args.mission_label,
            "site": args.site
        })
        controller = MissionController(
            mission_number=mission_num,
            direction=direction,
            with_mag=with_mag,
            expected_qr=expected_qr,
            bridge_host=args.bridge_host,
            enable_vision=True,
            enable_qr_check=False,
            enable_selector_check=True,
            enable_mag_check=True,
            bridge=bridge,
            auto_mode=True,
            mission_label=args.mission_label,
            request_id=args.request_id,
            site=args.site
        )
        try:
            controller.run()
        except BridgeAbort:
            sys.exit(2)
        except Exception:
            sys.exit(1)
        sys.exit(0)

    print("="*60)
    print("    비전 통합 미션 컨트롤러")
    print("="*60)

    def ask_yes_no(prompt: str, default: bool = True) -> bool:
        raw = input(prompt).strip().lower()
        if not raw:
            return default
        return raw.startswith('y')

    # 미션 번호 선택
    while True:
        try:
            mission_num = int(input("\n미션 번호를 선택하세요 (1 or 2): ").strip())
            if mission_num in (1, 2):
                break
            print("[ERROR] 1 또는 2를 입력하세요.")
        except ValueError:
            print("[ERROR] 숫자를 입력하세요.")
    
    # 방향 선택
    while True:
        direction = input("방향을 선택하세요 (in=불입, out=불출): ").strip().lower()
        if direction in ("in", "out"):
            break
        print("[ERROR] 'in' 또는 'out'을 입력하세요.")
    
    # 탄창 포함 여부
    with_mag_input = input("탄창 포함 여부 (y/n, 기본=y): ").strip().lower()
    with_mag = (with_mag_input != "n")
    
    # QR 코드 기대값 (불입 시에만 의미 있음)
    expected_qr = "2022120139"
    if direction == "in":
        qr_input = input(f"QR 코드 기대값 (기본={mission_num}): ").strip()
        if qr_input:
            expected_qr = qr_input
        else:
            expected_qr = str(mission_num)
    
    # 비전 활성화 여부
    enable_vision_input = input("비전 검사 활성화 (y/n, 기본=y): ").strip().lower()
    enable_vision = (enable_vision_input != "n")

    enable_qr_check = False
    enable_selector_check = True
    enable_mag_check = True

    # 브릿지 호스트 (옵션)
    bridge_host = input("브릿지 호스트 (기본=192.168.1.23): ").strip()
    if not bridge_host:
        bridge_host = "192.168.1.23"
    
    print(f"\n[설정]")
    print(f"  미션: {mission_num}번")
    print(f"  방향: {'불입(IN)' if direction == 'in' else '불출(OUT)'}")
    print(f"  탄창: {'포함' if with_mag else '미포함'}")
    if direction == "in":
        print(f"  QR 기대값: {expected_qr}")
    print(f"  비전 검사: {'활성화' if enable_vision else '비활성화'}")
    if enable_vision:
        print(f"    - QR 검사: {'실행' if enable_qr_check else '스킵'}")
        print(f"    - 조정간 검사: {'실행' if enable_selector_check else '스킵'}")
        print(f"    - 탄창 검사: {'실행' if enable_mag_check else '스킵'}")
    else:
        print("    - QR/조정간/탄창 검사: 스킵 (비전 비활성화)")
    print(f"  브릿지: {bridge_host}")
    
    confirm = input("\n시작하시겠습니까? (y/n): ").strip().lower()
    if confirm != "y":
        print("[중단] 사용자가 취소했습니다.")
        sys.exit(0)

    controller = MissionController(
        mission_number=mission_num,
        direction=direction,
        with_mag=with_mag,
        expected_qr=expected_qr,
        bridge_host=bridge_host,
        enable_vision=enable_vision,
        enable_qr_check=enable_qr_check,
        enable_selector_check=enable_selector_check,
        enable_mag_check=enable_mag_check,
        mission_label=None,
        request_id=None,
        bridge=BridgeInterface(enabled=False),
        auto_mode=False
    )
    
    controller.run()
