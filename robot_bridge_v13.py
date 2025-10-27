#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Robot Bridge (controller resident)  v1.3 (IK option wired, busy-safe, Actuator control)

- v1.2 기반(기존 기능 동일) + IK solver option 연동 추가
- 액추에이터 제어 명령 추가: actuator.forward, actuator.backward, actuator.home, actuator.status
- JSON 라인 프로토콜 추가 엔드포인트:
    * ik.set_option {option: { ... }}  → 기본 IK 옵션 저장 & 적용
    * ik.get_option                    → 현재 저장된 옵션 조회
- 모션(movej/movel/line/reljnt/relline)에서 per-command로 ik_option 전달 가능

주의
- 나머지 기능/엔드포인트는 v1.2 그대로 유지
"""

import json, socket, threading, traceback, time, os, glob, sys
from Queue import Queue, Empty  # Py2 환경

# --- 기존 i611 로봇 브릿지 의존 (원본 기반) ---
from i611_MCS import i611Robot, Position, Joint, MotionParam
from i611_io import dout, din, IOinit
from i611shm import shm_read
from rbsys import RobSys
from i611_extend import Pallet
import i611sys

# ---- pyserial import (안전 가드) ----
serial = None
_list_ports = None
try:
    import serial as _serial
    serial = _serial
    sys.stderr.write("[INIT] pyserial loaded (version: %s)\n" % getattr(_serial, "VERSION", "unknown"))
    sys.stderr.flush()
    try:
        from serial.tools import list_ports as _list_ports
        sys.stderr.write("[INIT] serial.tools.list_ports loaded\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write("[INIT] list_ports not available: %s\n" % e)
        sys.stderr.flush()
        _list_ports = None
except Exception as e:
    sys.stderr.write("[INIT] WARNING: pyserial not available: %s\n" % e)
    sys.stderr.write("[INIT] Install: pip install pyserial\n")
    sys.stderr.flush()
    serial = None
    _list_ports = None

HOST = "0.0.0.0"
PORT = 50050
RECV_TIMEOUT = 0.5
CLIENT_IDLE_MAX = 30

# SHM addrs
ADDR_BASE_BLOCK = 0x3000
ADDR_TCP        = 0x30E8
ADDR_JOINT      = 0x3050

# 서버측 안전 상/하한
LIM_JS = (1.0, 500.0)
LIM_LS = (1.0, 1500.0)
LIM_ACC = (0.05, 5.0)
LIM_OVERLAP = (0.0, 100.0)
DEFAULT_MIN_MOVE_MM = 20.0

ARDUINO_BAUD = 115200
ARDUINO_ID_EXPECT = "TB6600CTL"
PING_INTERVAL_SEC = 2.0

# 액추에이터 기본값
ACTUATOR_DEFAULT_DISTANCE = 800  # mm
ACTUATOR_DEFAULT_SPEED = 200     # mm/s
ACTUATOR_HOME_SPEED = 200        # mm/s

# ---------------- 공통 유틸 ----------------

def _float(v, default=None):
    try: return float(v)
    except: return default

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _apply_motionparam_from_cmd(rb, cmd):
    keys = ("jnt_speed","lin_speed","acctime","dacctime","pose_speed","overlap","speed")
    if not any(k in cmd for k in keys):
        return
    js = cmd.get("jnt_speed", None)
    ls = cmd.get("lin_speed", None)
    if "speed" in cmd:
        v = _float(cmd.get("speed"))
        if v is not None:
            if js is None: js = v
            if ls is None: ls = v
    mp = MotionParam()
    if js is not None: mp.jnt_speed = _clamp(_float(js, 20.0), LIM_JS[0], LIM_JS[1])
    if ls is not None: mp.lin_speed = _clamp(_float(ls, 100.0), LIM_LS[0], LIM_LS[1])
    if "acctime"   in cmd: mp.acctime   = _clamp(_float(cmd["acctime"], 0.25), LIM_ACC[0], LIM_ACC[1])
    if "dacctime"  in cmd: mp.dacctime  = _clamp(_float(cmd["dacctime"],0.25), LIM_ACC[0], LIM_ACC[1])
    if "pose_speed" in cmd: mp.pose_speed = _float(cmd["pose_speed"], 0.0)
    if "overlap"   in cmd: mp.overlap   = _clamp(_float(cmd["overlap"], 0.0), LIM_OVERLAP[0], LIM_OVERLAP[1])
    rb.motionparam(mp)

def _deg(x): return _float(x, 0.0) * 180.0 / 3.141592653589793
def _mm(x):  return _float(x, 0.0) * 1000.0

def _read_eval_block():
    info = shm_read(ADDR_BASE_BLOCK, 20).split(',')
    pos = [
        _mm(info[0]), _mm(info[1]), _mm(info[2]),
        _deg(info[3]), _deg(info[4]), _deg(info[5]),
        int(info[6]),
        int(info[8])
    ]
    jnt = [_deg(info[9]), _deg(info[10]), _deg(info[11]),
           _deg(info[12]), _deg(info[13]), _deg(info[14])]
    vel       = _float(info[17], 0.0)
    singular  = int(info[7])
    softlimit = int(info[19])
    return pos, jnt, vel, singular, softlimit

def _read_tcp_pose():
    info = shm_read(ADDR_TCP, 6).split(',')
    return [_mm(info[0]), _mm(info[1]), _mm(info[2]),
            _deg(info[3]), _deg(info[4]), _deg(info[5])] + [0,0]

def _to_mm_xyz(v):
    try:
        fv = float(v)
        if abs(fv) <= 2.0:
            return fv * 1000.0
        return fv
    except:
        return v

# ---------------- Arduino Bridge ----------------

class ArduinoBridge(object):
    """USB로 연결된 Arduino(TB6600CTL) 자동 탐지/연결 + 비동기 명령 처리"""
    def __init__(self, baud=ARDUINO_BAUD):
        sys.stderr.write("[Arduino] ArduinoBridge initializing (baud=%d)...\n" % baud)
        sys.stderr.flush()
        self.baud = baud
        self.ser = None
        self.port = None
        self.lock = threading.Lock()
        self.rx_buf = []  # 최근 수신 라인 저장
        self.stop_evt = threading.Event()
        # busy window
        self._busy = 0
        self._busy_lock = threading.Lock()
        self.worker = threading.Thread(target=self._maintain_loop)
        self.worker.daemon = True
        self.worker.start()
        sys.stderr.write("[Arduino] Background worker thread started\n")
        sys.stderr.flush()

    # --- busy helpers ---
    def _begin_busy(self):
        with self._busy_lock:
            self._busy += 1
    def _end_busy(self):
        with self._busy_lock:
            if self._busy > 0:
                self._busy -= 1
    def _is_busy(self):
        with self._busy_lock:
            return self._busy > 0

    # --- 내부: 디바이스 스캔 ---
    def _candidate_ports(self):
        sys.stderr.write("[Arduino] Scanning for ports...\n")
        sys.stderr.flush()
        pats = ["/dev/ttyACM*", "/dev/ttyUSB*"]
        lst = []
        for p in pats:
            lst.extend(glob.glob(p))
        if _list_ports is not None:
            try:
                for p in _list_ports.comports():
                    dev = getattr(p, 'device', None) or str(p)
                    if dev and dev not in lst:
                        lst.append(dev)
            except Exception:
                pass
        sys.stderr.write("[Arduino] Found %d candidate port(s): %s\n" % (len(lst), lst))
        sys.stderr.flush()
        return lst

    def _drain_banner(self, s, timeout=0.5):
        t0 = time.time()
        try:
            s.reset_input_buffer()
        except Exception:
            pass
        while time.time() - t0 < timeout:
            line = s.readline()
            if not line:
                break

    def _try_open(self, dev):
        sys.stderr.write("[Arduino] Trying port: %s\n" % dev)
        sys.stderr.flush()
        if serial is None:
            sys.stderr.write("[Arduino] ERROR: pyserial not loaded\n")
            sys.stderr.flush()
            return None
        try:
            # timeout 살짝 늘리고, 오픈 직후 리셋 대기 보장
            s = serial.Serial(dev, self.baud, timeout=0.8)
            time.sleep(1.8)
        except Exception:
            sys.stderr.write("[Arduino] Failed to open %s\n" % dev)
            sys.stderr.flush()
            return None
        try:
            # 배너는 버리지 않고, 바로 ID?를 전송하여 응답 대기
            s.write(b"ID?\n"); s.flush()
            deadline = time.time() + 1.2
            ok = False
            while time.time() < deadline:
                line = s.readline()
                if not line:
                    continue
                txt = None
                try:
                    txt = line.decode('utf-8', 'ignore').strip()
                except Exception:
                    txt = str(line).strip()
                sys.stderr.write("[Arduino] %s responded: %s\n" % (dev, txt))
                sys.stderr.flush()
                if txt.startswith("ID:") and ARDUINO_ID_EXPECT in txt:
                    ok = True; break
                if txt == "PONG":
                    ok = True; break
            if not ok:
                s.write(b"PING\n"); s.flush()
                deadline = time.time() + 1.2
                while time.time() < deadline:
                    line = s.readline()
                    if not line:
                        continue
                    try:
                        if line.decode('utf-8','ignore').strip() == "PONG":
                            ok = True; break
                    except Exception:
                        pass
            if ok:
                return s
        except Exception:
            pass
        try:
            s.close()
        except Exception:
            pass
        return None

    def _maintain_loop(self):
        last_ping = time.time()
        while not self.stop_evt.is_set():
            if self.ser is None:
                for dev in self._candidate_ports():
                    s = self._try_open(dev)
                    if s:
                        with self.lock:
                            self.ser = s; self.port = dev
                        sys.stderr.write("[arduino] connected %s\n" % dev)
                        last_ping = time.time()
                        break
                if self.ser is None:
                    sys.stderr.write("[Arduino] No device connected, waiting...\n")
                    sys.stderr.flush()
                    time.sleep(1.0)
                    continue
            try:
                now = time.time()
                if now - last_ping >= PING_INTERVAL_SEC:
                    if not self._is_busy():
                        # --- 개선된 PONG 대기: 여러 라인을 무시/소화하며 PONG을 찾는다 ---
                        with self.lock:
                            self.ser.write(b"PING\n")
                            self.ser.flush()
                        deadline = time.time() + 1.5   # 여유 있는 대기
                        pong_ok = False
                        while time.time() < deadline:
                            line = self.ser.readline()
                            if not line:
                                continue
                            try:
                                txt = line.decode('utf-8','ignore').strip()
                            except Exception:
                                txt = str(line).strip()
                            if txt == 'PONG':
                                pong_ok = True
                                break
                            # 다른 진단/배너 라인은 캐시에 쌓고 계속 PONG을 기다림
                            if txt:
                                with self.lock:
                                    self.rx_buf.append(txt)
                                    if len(self.rx_buf) > 200:
                                        self.rx_buf = self.rx_buf[-200:]
                        if not pong_ok:
                            raise IOError("no PONG")
                    last_ping = now
                line = self.ser.readline()

                if line:
                    try:
                        txt = line.decode('utf-8', 'ignore').rstrip('\r\n')
                    except Exception:
                        txt = str(line).rstrip()
                    if txt:
                        with self.lock:
                            self.rx_buf.append(txt)
                            if len(self.rx_buf) > 200:
                                self.rx_buf = self.rx_buf[-200:]
                else:
                    time.sleep(0.01)
            except Exception as e:
                sys.stderr.write("[arduino] lost: %s\n" % e)
                try:
                    with self.lock:
                        if self.ser:
                            try: self.ser.close()
                            except: pass
                        self.ser = None; self.port = None
                except: pass
                time.sleep(0.5)

    # --- 공용 API ---
    def status(self):
        with self.lock:
            return {
                "connected": self.ser is not None,
                "port": self.port,
                "rx_cached": len(self.rx_buf),
                "last": (self.rx_buf[-1] if self.rx_buf else None)
            }

    def flush(self):
        with self.lock:
            if self.ser:
                try: self.ser.reset_input_buffer()
                except: pass
            self.rx_buf[:] = []
        return {"ok": True}

    def readline(self, timeout=0.0):
        t0 = time.time()
        while True:
            with self.lock:
                if self.rx_buf:
                    return self.rx_buf.pop(0)
            if timeout <= 0 or time.time() - t0 >= timeout:
                return None
            time.sleep(0.01)

    def exec_line(self, line, collect_ms=300, overall_timeout=3.0):
        self._begin_busy()
        try:
            with self.lock:
                if not self.ser:
                    raise RuntimeError("Arduino not connected")
                data = (line.strip() + "\n").encode('utf-8')
                self.ser.write(data); self.ser.flush()
            out = []
            t0 = time.time(); last = time.time()
            while True:
                ln = self.readline(timeout=0.05)
                if ln is not None:
                    out.append(ln); last = time.time()
                if (time.time() - last) * 1000.0 >= collect_ms:
                    break
                if time.time() - t0 > overall_timeout:
                    break
            return out
        finally:
            self._end_busy()

    def shutdown(self):
        self.stop_evt.set()
        try:
            if self.ser:
                self.ser.close()
        except: pass

# ---------------- Robot Bridge ----------------

class Bridge(object):
    def __init__(self):
        IOinit()
        self.rbs = RobSys()
        try:
            self.rbs.open()
            try: self.rbs.cmd_stop()
            except: pass
            try: self.rbs.cmd_reset()
            except: pass
        except: pass

        self.rb = None
        self.rb_opened = False
        self._rb_ready_req  = threading.Event()
        self._rb_ready_done = threading.Event()
        self._rb_ready_err  = None

        # 로봇 명령 전용 큐/워커
        self.jobq = Queue()
        self.stop_evt  = threading.Event()
        self.pause_evt = threading.Event()

        self.pallet = None

        self.worker = threading.Thread(target=self._worker_loop)
        self.worker.daemon = True
        self.worker.start()

        # 메모리 IO 단일비트 바인딩
        self._io_bit_read  = None
        self._io_bit_write = None
        self.sys = None
        try:
            if hasattr(i611sys, "read_memory_io_bit") and hasattr(i611sys, "write_memory_io_bit"):
                self._io_bit_read  = i611sys.read_memory_io_bit
                self._io_bit_write = i611sys.write_memory_io_bit
            else:
                for clsname in ("i611sys", "RobSys", "SysIO"):
                    cls = getattr(i611sys, clsname, None)
                    if cls is None: continue
                    try:
                        obj = cls()
                        if hasattr(obj, "read_memory_io_bit") and hasattr(obj, "write_memory_io_bit"):
                            self.sys = obj
                            self._io_bit_read  = obj.read_memory_io_bit
                            self._io_bit_write = obj.write_memory_io_bit
                            break
                    except: pass
        except:
            self._io_bit_read = self._io_bit_write = None

        # === 추가: Arduino 브릿지 ===
        self.ardu = ArduinoBridge()
        sys.stderr.write("[INIT] Creating ArduinoBridge instance...\n")
        sys.stderr.flush()

        # === 추가: IK 옵션 기본값 저장 ===
        self._ik_default = None  # dict or None

    def _get_sysport(self):
        return i611Robot.get_system_port()

    def _request_rb_ready_on_main(self, timeout=5.0):
        if self.rb is not None and self.rb_opened: return
        self._rb_ready_err = None
        self._rb_ready_done.clear()
        self._rb_ready_req.set()
        t0 = time.time()
        while not self._rb_ready_done.is_set():
            if time.time() - t0 > timeout:
                raise RuntimeError("i611Robot ready 대기 타임아웃")
            time.sleep(0.01)
        if self._rb_ready_err:
            raise RuntimeError("i611Robot 준비 실패: %s" % self._rb_ready_err)

    def _ensure_ready_and_params(self, cmd):
        st = self._get_sysport()
        svon, emo, error = int(st[1]), int(st[2]), int(st[7])
        if emo != 0:
            raise RuntimeError("EMO 활성화. 해제 후 시도. st=%s" % (st,))
        if error != 0:
            try:
                if self.rbs:
                    self.rbs.cmd_reset(); time.sleep(0.2)
                    st = self._get_sysport(); error = int(st[7])
            except: pass
        if error != 0:
            raise RuntimeError("시스템 에러 잔존. st=%s" % (st,))
        if svon != 1:
            raise RuntimeError("Servo OFF. st=%s" % (st,))
        self._request_rb_ready_on_main()
        _apply_motionparam_from_cmd(self.rb, cmd)
        # 모션 전 기본 IK 옵션 적용
        self._apply_ik_option(self._ik_default)

    # === IK 적용 유틸 ===
    def _apply_ik_option(self, opt):
        """opt(dict)를 드라이버에 적용. 가능한 함수명을 순차 시도."""
        if not opt:
            return False
        ok = False
        for name in ("ik_solver_option", "ik_set_option", "set_ik_solver_option", "setikoption"):
            fn = getattr(self.rb, name, None)
            if not callable(fn):
                continue
            # 딕셔너리 통째 전달
            try:
                fn(opt); ok = True; break
            except Exception:
                pass
            # kwargs로 전달
            try:
                fn(**opt); ok = True; break
            except Exception:
                continue
        return ok

    class _IKTempCtx(object):
        """per-command ik_option 임시 적용용 컨텍스트"""
        def __init__(self, bridge, temp_opt):
            self.bridge = bridge
            self.temp = temp_opt
            self.saved = bridge._ik_default
            self.applied = False
        def __enter__(self):
            if self.temp:
                self.bridge._apply_ik_option(self.temp)
                self.applied = True
            return self
        def __exit__(self, exc_type, exc, tb):
            # 기본값으로 복원
            if self.applied and self.saved is not None:
                try:
                    self.bridge._apply_ik_option(self.saved)
                except Exception:
                    pass

    def _worker_loop(self):
        while not self.stop_evt.is_set():
            try:
                cmd = self.jobq.get(timeout=0.2)
            except Empty:
                continue
            try:
                while self.pause_evt.is_set() and not self.stop_evt.is_set():
                    time.sleep(0.05)
                if self.stop_evt.is_set(): break
                self._exec_cmd(cmd)
            except Exception:
                try: cmd["_send"]({"ok": False, "err": traceback.format_exc()})
                except: pass
            finally:
                self.jobq.task_done()

    # ============== 명령 처리 ==============
    def _exec_cmd(self, cmd):
        name  = cmd.get("cmd")
        reply = cmd.get("_send")

        # ---- Arduino ----
        if name == "arduino.stat":
            return reply(dict({"ok": True}, **self.ardu.status()))
        if name == "arduino.flush":
            return reply(self.ardu.flush())
        if name == "arduino.readline":
            ln = self.ardu.readline(timeout=_float(cmd.get("timeout",0.0),0.0) or 0.0)
            return reply({"ok": True, "line": ln})
        if name == "arduino.exec":
            line = cmd.get("line", "")
            if not isinstance(line, basestring) or not line.strip():
                return reply({"ok": False, "err": "line required"})
            try:
                lines = self.ardu.exec_line(line,
                                            collect_ms=int(cmd.get("collect_ms",300)),
                                            overall_timeout=_float(cmd.get("timeout",3.0),3.0))
                return reply({"ok": True, "lines": lines})
            except Exception as e:
                return reply({"ok": False, "err": str(e)})

        # ---- 액추에이터 제어 (간소화 명령) ----
        if name == "actuator.forward":
            # 전진 명령 - 컨트롤러에서 M1 명령 생성
            distance = _float(cmd.get("distance"), ACTUATOR_DEFAULT_DISTANCE)
            speed = _float(cmd.get("speed"), ACTUATOR_DEFAULT_SPEED)
            
            try:
                arduino_cmd = "M1 {0} {1}".format(distance, speed)
                lines = self.ardu.exec_line(
                    arduino_cmd,
                    collect_ms=int(cmd.get("collect_ms", 300)),
                    overall_timeout=_float(cmd.get("timeout", 5.0), 5.0)
                )
                # 이동 예상 시간 동안 busy 유지 → 백그라운드 PING 차단
                wait_time = max(0.0, abs(distance) / max(1e-6, speed) + 0.5)
                try:
                    self.ardu._begin_busy()
                    time.sleep(wait_time)
                finally:
                    self.ardu._end_busy()
                
                return reply({"ok": True, "lines": lines, "distance": distance, "speed": speed, "direction": "forward"})
            except Exception as e:
                return reply({"ok": False, "err": str(e)})
        
        if name == "actuator.backward":
            # 후진 명령 - 컨트롤러에서 M1 명령 생성
            distance = _float(cmd.get("distance"), ACTUATOR_DEFAULT_DISTANCE)
            speed = _float(cmd.get("speed"), ACTUATOR_DEFAULT_SPEED)
            
            try:
                arduino_cmd = "M1 -{0} {1}".format(distance, speed)  # 음수로 후진
                lines = self.ardu.exec_line(
                    arduino_cmd,
                    collect_ms=int(cmd.get("collect_ms", 300)),
                    overall_timeout=_float(cmd.get("timeout", 5.0), 5.0)
                )
                wait_time = max(0.0, abs(distance) / max(1e-6, speed) + 0.5)
                try:
                    self.ardu._begin_busy()
                    time.sleep(wait_time)
                finally:
                    self.ardu._end_busy()
                
                return reply({"ok": True, "lines": lines, "distance": distance, "speed": speed, "direction": "backward"})
            except Exception as e:
                return reply({"ok": False, "err": str(e)})
        
        if name == "actuator.home":
            # 원점 복귀 - 아두이노 HOME 명령
            try:
                lines = self.ardu.exec_line(
                    "HOME",
                    collect_ms=int(cmd.get("collect_ms", 400)),
                    overall_timeout=_float(cmd.get("timeout", 8.0), 8.0)
                )
                # 홈 직후 안정 시간 동안 busy 유지
                try:
                    self.ardu._begin_busy()
                    time.sleep(0.5)
                finally:
                    self.ardu._end_busy()
                return reply({
                    "ok": True,
                    "lines": lines,
                    "action": "home"
                })
            except Exception as e:
                return reply({"ok": False, "err": str(e)})

        
        if name == "actuator.status":
            # 액추에이터/아두이노 상태 확인
            try:
                ardu_stat = self.ardu.status()
                return reply({"ok": True, "arduino": ardu_stat})
            except Exception as e:
                return reply({"ok": False, "err": str(e)})

        # ---- 진단/관리 ----
        if name == "svstat":
            return reply({"ok": True, "st": self._get_sysport()})

        if name == "reset":
            try:
                if self.rbs:
                    try: self.rbs.cmd_stop()
                    except: pass
                    try: self.rbs.cmd_reset()
                    except: pass
            except Exception as e:
                return reply({"ok": False, "err": "reset failed: %s" % e})
            return reply({"ok": True, "st": self._get_sysport()})

        if name == "where":
            jtxt = shm_read(ADDR_JOINT, 6)
            ptxt = shm_read(ADDR_BASE_BLOCK, 6)
            return reply({"ok": True, "joint": jtxt, "pose": ptxt})

        if name == "where_conv":
            pos, jnt, vel, singular, softlimit = _read_eval_block()
            return reply({"ok": True, "pose": pos, "joint": jnt,
                          "vel": vel, "singular": singular, "softlimit": softlimit})

        if name == "tcp":
            tcp = _read_tcp_pose()
            return reply({"ok": True, "tcp": tcp})

        if name == "pause":
            self.pause_evt.set();  return reply({"ok": True})
        if name == "continue":
            self.pause_evt.clear(); return reply({"ok": True})
        if name == "stop":
            while True:
                try: self.jobq.get_nowait(); self.jobq.task_done()
                except Empty: break
            try:
                if self.rb_opened and self.rb: self.rb.stop()
            except: pass
            return reply({"ok": True})

        # ---- IK 옵션 ----
        if name == "ik.set_option":
            opt = cmd.get("option") or {}
            if not isinstance(opt, dict):
                return reply({"ok": False, "err": "option must be dict"})
            self._request_rb_ready_on_main()
            # 저장 + 즉시 적용
            self._ik_default = dict(opt)
            applied = self._apply_ik_option(self._ik_default)
            return reply({"ok": True, "applied": bool(applied), "option": self._ik_default})

        if name == "ik.get_option":
            return reply({"ok": True, "option": (self._ik_default or {})})

        # ---- 모션/IO ----
        if name in ("movej","movel","line","relline","reljnt"):
            self._ensure_ready_and_params(cmd)
            ik_opt = cmd.get("ik_option") if isinstance(cmd.get("ik_option"), dict) else None
            with self._IKTempCtx(self, ik_opt):
                if name == "movej":
                    # accept both 'j' (legacy) and 'joint' (SDK v1.3)
                    j = cmd.get("j")
                    if j is None:
                        j = cmd.get("joint") or []
                    self.rb.move(Joint(j))
                    return reply({"ok": True})
                if name == "movel":
                    # accept both 'p' and 'pose'
                    p = cmd.get("p")
                    if p is None:
                        p = cmd.get("pose") or []
                    self.rb.move(Position(p[0],p[1],p[2],p[3],p[4],p[5]))
                    return reply({"ok": True})
                if name == "line":
                    # accept both 'p' and 'pose'
                    p = cmd.get("p")
                    if p is None:
                        p = cmd.get("pose") or []
                    self.rb.line(Position(p[0],p[1],p[2],p[3],p[4],p[5]))
                    return reply({"ok": True})
                if name == "relline":
                    # accept both 'dp' and 'delta_p'
                    dp = cmd.get("dp")
                    if dp is None:
                        dp = cmd.get("delta_p") or [0,0,0,0,0,0]
                    dx,dy,dz,drz,dry,drx = dp
                    dx,dy,dz = _to_mm_xyz(dx), _to_mm_xyz(dy), _to_mm_xyz(dz)
                    confirm_large = bool(cmd.get("confirm_large", False))
                    if not confirm_large:
                        if max(abs(float(dx)), abs(float(dy)), abs(float(dz))) > DEFAULT_MIN_MOVE_MM:
                            raise RuntimeError("relline too large (>%.1f mm). confirm_large=true" % DEFAULT_MIN_MOVE_MM)
                    self.rb.relline(float(dx), float(dy), float(dz), float(drz), float(dry), float(drx))
                    return reply({"ok": True})
                if name == "reljnt":
                    # accept both 'dj' and 'delta_j'
                    dj = cmd.get("dj")
                    if dj is None:
                        dj = cmd.get("delta_j") or [0,0,0,0,0,0]
                    confirm_large = bool(cmd.get("confirm_large", False))
                    if not confirm_large:
                        if max([abs(_float(x, 0.0)) for x in dj]) > 10.0:
                            raise RuntimeError("reljnt too large (>10 deg). confirm_large=true")
                    self.rb.reljntmove(float(dj[0]),float(dj[1]),float(dj[2]),
                                       float(dj[3]),float(dj[4]),float(dj[5]))
                    return reply({"ok": True})

        if name == "io":
            mode = cmd.get("mode","get")
            if mode == "set":
                self._ensure_ready_and_params(cmd)
                start = int(cmd.get("start",16))
                bits  = cmd.get("bits","")
                dout(start, bits)
                return reply({"ok": True})
            elif mode == "set_bit":
                # SDK v1.3 경로: 단일 비트 토글
                idx = int(cmd.get("index"))
                val = 1 if int(cmd.get("val",1)) else 0
                try:
                    # 우선 메모리 IO 단일비트 API가 있으면 사용
                    if self._io_bit_write:
                        self._io_bit_write(idx, val)
                    else:
                        # 없으면 dout로 해당 비트만 세트 (컨트롤러가 지원하는 단위에 맞춤)
                        dout(idx, "1" if val else "0")
                except Exception as e:
                    return reply({"ok": False, "err": "io.set_bit failed: %s" % e})
                return reply({"ok": True})
            else:
                adr = int(cmd.get("adr",0)); width = int(cmd.get("width",1))
                val = din(adr, width)
                return reply({"ok": True, "val": int(val)})

        if name == "io.wait":
            adr = int(cmd.get("adr",0))
            target = int(cmd.get("target",1))
            timeout = _float(cmd.get("timeout",5.0), 5.0)
            t0 = time.time()
            while True:
                v = 1 if (int(din(adr, 1)) & 0x01) != 0 else 0
                if v == target: return reply({"ok": True, "val": v})
                if time.time() - t0 > timeout:
                    return reply({"ok": False, "err": "io.wait timeout", "val": v})
                time.sleep(0.01)

        # ---- 모션 파라미터 ----
        if name == "get_motion":
            self._ensure_ready_and_params({})
            lst = self.rb.getmotionparam().mp2list()
            return reply({"ok": True, "mp": lst})
        if name == "set_motion":
            self._ensure_ready_and_params(cmd)
            return reply({"ok": True})

        # ---- 팔레트 ----
        if name == "pallet.clear":
            self.pallet = None; return reply({"ok": True})
        if name == "pallet.init_4":
            self._ensure_ready_and_params({})
            p0,p1,p2,p3 = cmd.get("p0"),cmd.get("p1"),cmd.get("p2"),cmd.get("p3")
            i = int(cmd.get("i",1)); j = int(cmd.get("j",1))
            top_first = int(cmd.get("top_first",1))
            self.pallet = Pallet()
            self.pallet.init_4(Position(*p0),Position(*p1),Position(*p2),Position(*p3),
                               i=i, j=j, top_first=top_first)
            return reply({"ok": True})
        if name == "pallet.adjust":
            if not self.pallet: return reply({"ok": False, "err":"pallet not initialized"})
            i = int(cmd.get("i",0)); j = int(cmd.get("j",0))
            dx = _float(cmd.get("dx",0.0),0.0); dy = _float(cmd.get("dy",0.0),0.0)
            self.pallet.adjust(i,j,dx,dy)
            return reply({"ok": True})
        if name == "pallet.cell":
            if not self.pallet: return reply({"ok": False, "err":"pallet not initialized"})
            i = int(cmd.get("i",0)); j = int(cmd.get("j",0))
            pos = self.pallet.get_cell(i,j)
            p = [pos.x,pos.y,pos.z,pos.rz,pos.ry,pos.rx]
            return reply({"ok": True, "p": p})

        # ---- IO WORD / BLOCK ----
        if name == "io.word":
            adr = int(cmd.get("adr",0)); width = int(cmd.get("width",32))
            val = din(adr, width)
            return reply({"ok": True, "adr": adr, "width": width, "val": int(val)})

        if name == "io.block":
            lst = cmd.get("adrs") or []; width = int(cmd.get("width",32))
            out = {}
            for a in lst:
                v = din(int(a), width); out[str(a)] = int(v)
            return reply({"ok": True, "width": width, "vals": out})

        # ---- 메모리 IO 단일비트 ----
        if name == "io.bit.get":
            port = int(cmd.get("port", 0))
            try:
                if self._io_bit_read:
                    val = self._io_bit_read(port)
                else:
                    val = 1 if (int(din(port, 1)) & 0x01) != 0 else 0
                return reply({"ok": True, "port": port, "val": int(val)})
            except Exception as e:
                return reply({"ok": False, "err": "io.bit.get failed: %s" % e})

        if name == "io.bit.set":
            port = int(cmd.get("port", 0))
            val  = 1 if int(cmd.get("val", 1)) else 0
            try:
                if self._io_bit_write:
                    self._io_bit_write(port, val)
                else:
                    dout(port, "1" if val else "0")
                return reply({"ok": True})
            except Exception as e:
                return reply({"ok": False, "err": "io.bit.set failed: %s" % e})

        # ---- Teach 준비 ----
        if name == "teach":
            try:
                import zeusteach
                zeusteach.ZeusTeach()
                return reply({"ok": True, "note": "ZeusTeach ready"})
            except Exception as e:
                return reply({"ok": False, "err": str(e)})

        return reply({"ok": False, "err": "unknown cmd: %s" % name})

    def enqueue(self, req, send_fn):
        req2 = dict(req); req2["_send"] = send_fn
        self.jobq.put(req2)

    def shutdown(self):
        self.stop_evt.set(); self.pause_evt.clear()
        try:
            if self.rb_opened and self.rb: self.rb.stop()
        except: pass
        try:
            if self.rbs: self.rbs.close()
        except: pass
        try:
            if self.ardu: self.ardu.shutdown()
        except: pass

    # 메인 스레드 펌프: i611Robot 생성/open 전용
    def run_main_pump(self):
        while not self.stop_evt.is_set():
            if self._rb_ready_req.is_set():
                self._rb_ready_req.clear(); self._rb_ready_err = None
                try:
                    if self.rb is None: self.rb = i611Robot()
                    if not self.rb_opened:
                        self.rb.open(); self.rb_opened = True
                except Exception as e:
                    self._rb_ready_err = str(e)
                finally:
                    self._rb_ready_done.set()
            time.sleep(0.01)

class JSONServer(object):
    def __init__(self, bridge):
        self.bridge = bridge
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((HOST, PORT)); self.sock.listen(1)

    def serve_forever(self):
        while True:
            conn, addr = self.sock.accept()
            try:
                conn.settimeout(RECV_TIMEOUT)
                self._handle_client(conn, addr)
            except Exception:
                try: conn.close()
                except: pass

    def _handle_client(self, conn, addr):
        fp = conn.makefile("rwb"); last = time.time()
        def send(obj):
            try: data = (json.dumps(obj) + "\n").encode("utf-8")
            except Exception: data = (json.dumps(obj) + "\n")
            fp.write(data); fp.flush()
        try:
            send({"ok": True, "hello": "robot_bridge", "version": "1.3-actuator"})
            while True:
                if time.time() - last > CLIENT_IDLE_MAX:
                    send({"ok": False, "err": "idle timeout"}); break
                try:
                    line = fp.readline()
                    if not line: break
                    last = time.time()
                    try: txt = line.decode("utf-8")
                    except Exception: txt = line
                    req = json.loads((txt.strip() or "{}"))

                    # 즉시 처리(로봇 큐 우회) — 상태/진단/IO/아두이노/IK 조회/설정/액추에이터
                    if req.get("cmd") in (
                        "stop","pause","continue","where","where_conv","tcp",
                        "svstat","reset","teach","get_motion",
                        "io","io.wait","io.word","io.block",
                        "pallet.clear","pallet.cell",
                        # Arduino
                        "arduino.stat","arduino.exec","arduino.readline","arduino.flush",
                        # IK 옵션
                        "ik.set_option","ik.get_option",
                        # 액추에이터
                        "actuator.forward","actuator.backward","actuator.home","actuator.status",
                    ):
                        self.bridge._exec_cmd(dict(req, _send=send))
                    else:
                        # 로봇 동작은 직렬 큐로 보냄
                        self.bridge.enqueue(req, send)
                except socket.timeout:
                    continue
                except ValueError as e:
                    send({"ok": False, "err": "bad json: %s" % e})
                except Exception:
                    send({"ok": False, "err": traceback.format_exc()})
        finally:
            try: fp.close()
            except: pass
            try: conn.close()
            except: pass

import sys
sys.stderr.write("="*60 + "\n")
sys.stderr.write("Robot Bridge (controller resident)  v1.3\n")
sys.stderr.write("(IK option wired, busy-safe, Actuator control)\n")
sys.stderr.write("="*60 + "\n")
sys.stderr.flush()

if __name__ == "__main__":
    br = Bridge()
    sys.stderr.write("[INIT] Bridge instance created\n")
    sys.stderr.flush()
    srv = JSONServer(br)
    sys.stderr.write("[INIT] JSON Server created, listening on %s:%d\n" % (HOST, PORT))
    sys.stderr.flush()
    t = threading.Thread(target=srv.serve_forever); t.daemon = True; t.start()
    sys.stderr.write("[INIT] Server thread started\n")
    sys.stderr.flush()
    try: br.run_main_pump()
    finally: br.shutdown()