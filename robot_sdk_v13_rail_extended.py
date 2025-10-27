# -*- coding: utf-8 -*-
"""
rb_sdk v1.3 (PC-side SDK for robot_bridge v1.3) - Extended with Linear Rail Control
- Keeps v12 API fully compatible
- Adds IK solver option wiring:
    * BridgeClient.ik_set_option(option: dict) -> {'ok': True, ...}
    * BridgeClient.ik_get_option() -> {'ok': True, 'option': {...}}
    * Per-motion call can pass ik_option=dict(...), forwarded to bridge
- Minor: hello() now returns bridge hello_info (no command roundtrip)
- Extended: Linear Rail (Actuator) Control Methods added
"""

import socket, json, time
from contextlib import contextmanager
from math import copysign

# --- Py2/3 compatibility shims ---
try:
    basestring   # Python 2
except NameError:
    basestring = str  # Python 3

def _as_text(x):
    if isinstance(x, bytes):
        try:
            return x.decode('utf-8', 'ignore')
        except Exception:
            return str(x)
    return x

class BridgeError(RuntimeError): pass
class Timeout(BridgeError): pass

class Profiles:
    TEACH = dict(jnt_speed=10.0, lin_speed=50.0,  acctime=0.30, dacctime=0.30, overlap=0.0)
    SAFE  = dict(jnt_speed=20.0, lin_speed=100.0, acctime=0.25, dacctime=0.25, overlap=0.0)
    FAST  = dict(jnt_speed=40.0, lin_speed=250.0, acctime=0.20, dacctime=0.20, overlap=10.0)

def _apply_profile_and_merge(kwargs, profile):
    base = {}
    if profile:
        if isinstance(profile, basestring):
            base = getattr(Profiles, profile.upper())
        elif isinstance(profile, dict):
            base = profile
    merged = dict(base)
    merged.update(kwargs or {})
    if "speed" in merged and merged["speed"] is not None:
        s = float(merged.pop("speed"))
        merged.setdefault("jnt_speed", s)
        merged.setdefault("lin_speed", s)
    return merged

def parse_joint_csv(txt):
    if not txt: return []
    return [float(x.strip()) for x in txt.split(',') if x.strip() != '']

def parse_pose_csv(txt):
    if not txt: return []
    return [float(x.strip()) for x in txt.split(',') if x.strip() != '']


class BridgeClient(object):
    # Defaults
    HOME_JSON_PATH   = "./home_pose.json"
    DEFAULT_PROFILE  = "TEACH"
    LIN_STEP_MM      = 2.0
    ANG_STEP_DEG     = 2.0
    RELJ_STEP_DEG    = 5.0

    # Electric gripper memory IO bits (close=48, open=50 by default)
    GRIPPER_DO1 = 48
    GRIPPER_DO3 = 50

    # Pneumatic DO map (SUCK=27, BLOW=29 by default)
    PNEU_SUCK_DO = 27
    PNEU_BLOW_DO = 29

    ZERO_ENCODER_JOINT = [-0.01, 0.0, 180.0, -0.02, 0.14, -0.03]
    
    # Linear Rail (Actuator) Defaults
    RAIL_MAX_STROKE = 800.0  # mm
    RAIL_DEFAULT_SPEED = 250.0  # mm/s
    RAIL_HOME_SPEED = 250.0  # mm/s

    def __init__(self, host="192.168.1.23", port=50050, connect_timeout=3.0, io_timeout=8.0, retry=0):
        self.host = host; self.port = port
        self.connect_timeout = connect_timeout
        self.io_timeout = io_timeout
        self.retry = retry
        self._sock = None; self._f = None
        self.hello_info = None
        self._rail_position = 0.0  # 현재 레일 위치 추적 (mm)

    # ---- Session ----
    def connect(self):
        s = socket.create_connection((self.host, self.port), self.connect_timeout)
        s.settimeout(self.io_timeout)
        f = s.makefile("rwb")
        hello = self._readline_json(f)
        if not (isinstance(hello, dict) and hello.get("ok") and hello.get("hello") == "robot_bridge"):
            raise BridgeError("Bridge hello failed: %r" % (hello,))
        self._sock, self._f = s, f
        self.hello_info = hello
        # --- 연결 직후 레일 위치를 실제 하드웨어에서 동기화 ---
        try:
            self._sync_rail_position_from_arduino()
        except Exception:
            # 동기화 실패해도 연결 자체는 유지
            pass
        return hello

    def hello(self):
        """Return bridge hello info captured at connect()."""
        return self.hello_info or {}

    def close(self):
        try:
            if self._f: self._f.close()
        finally:
            self._f = None
            if self._sock:
                try: self._sock.close()
                except: pass
            self._sock = None

    def _readline_json(self, f):
        line = f.readline()
        if not line: raise BridgeError("Disconnected")
        try: line = line.decode("utf-8")
        except Exception: pass
        return json.loads((line.strip() or "{}"))

    def _send_once(self, obj):
        data = (json.dumps(obj) + "\n").encode("utf-8")
        prev_to = None
        try:
            # arduino.exec can take longer; propagate timeout to socket temporarily
            if self._sock and obj.get("cmd") == "arduino.exec":
                req_to = float(obj.get("timeout", 0.0) or 0.0)
                if req_to and (self._sock.gettimeout() or 0.0) < req_to + 2.0:
                    prev_to = self._sock.gettimeout()
                    self._sock.settimeout(req_to + 2.0)
            self._f.write(data); self._f.flush()
            resp = self._readline_json(self._f)
        finally:
            if prev_to is not None and self._sock:
                self._sock.settimeout(prev_to)
        if not isinstance(resp, dict):
            raise BridgeError("Bad response: %r" % (resp,))
        if not resp.get("ok", False) and "err" in resp:
            raise BridgeError(resp.get("err"))
        return resp

    def _send_cmd(self, obj):
        if not self._f: raise BridgeError("Not connected")
        trials = self.retry + 1; last = None
        for i in range(trials):
            try: return self._send_once(obj)
            except (socket.timeout, BridgeError) as e:
                last = e
                if i < trials - 1: time.sleep(0.1); continue
                raise
        raise last or BridgeError("Unknown send error")

    def __enter__(self): self.connect(); return self
    def __exit__(self, *exc): self.close()
    @contextmanager
    def session(self):
        try:
            self.connect(); yield self
        finally:
            self.close()

    # ---------------- Diagnostics/Control ----------------
    def svstat(self): return self._send_cmd({"cmd":"svstat"})["st"]
    def reset(self):  return self._send_cmd({"cmd":"reset"})["st"]
    def where_raw(self):
        r = self._send_cmd({"cmd":"where"}); return r["joint"], r["pose"]
    def where_conv(self):
        r = self._send_cmd({"cmd":"where_conv"})
        return r["pose"], r["joint"], r.get("singular",0), r.get("softlimit",0), r.get("vel",0.0)
    def tcp(self): return self._send_cmd({"cmd":"tcp"})["tcp"]
    def stop(self):  return self._send_cmd({"cmd":"stop"})
    def pause(self): return self._send_cmd({"cmd":"pause"})
    def cont(self):  return self._send_cmd({"cmd":"continue"})
    def teach_ready(self): return self._send_cmd({"cmd":"teach"})

    # ---- IK option wiring (bridge v1.3) ----
    def ik_set_option(self, option):
        """option: dict to set default IK preference on bridge (persist for subsequent motions)."""
        if not isinstance(option, dict):
            raise ValueError("ik_set_option requires dict")
        return self._send_cmd({"cmd":"ik.set_option", "option": dict(option)})

    def ik_get_option(self):
        return self._send_cmd({"cmd":"ik.get_option"}).get("option", {})

    # ---------------- I/O ----------------
    def dio_set(self, start, bits):
        """
        i611_io.dout(start, bits) 래퍼.
        bits는 '0101...' 문자열. 여러 비트를 연속으로 갱신할 때만 사용.
        단일 비트 토글의 기본 경로는 dio_set_bit()를 사용한다.
        """
        return self._send_cmd({"cmd":"io","mode":"set","start":int(start),"bits":str(bits)})

    def dio_set_bit(self, index, val):
        """
        단일 비트 전용. bridge: {cmd:'io', mode:'set_bit', index:<int>, val:<0|1>}
        """
        return self._send_cmd({"cmd":"io","mode":"set_bit","index":int(index),"val":1 if val else 0})
    
    def dio_get(self, adr, width=1):
        r = self._send_cmd({"cmd":"io","mode":"get","adr":int(adr),"width":int(width)})
        v = r.get("val", 0)
        try: return int(v)
        except Exception:
            s = str(v).strip(); return int(s) if s else 0
    def dio_wait(self, adr, target=1, timeout=5.0):
        try:
            r = self._send_cmd({"cmd":"io.wait","adr":int(adr),"target":int(target),"timeout":float(timeout)})
        except BridgeError as e:
            if "timeout" in str(e).lower(): raise Timeout(str(e))
            raise
        return r.get("val", target)

    def io_word(self, adr, width=32):
        return self._send_cmd({"cmd":"io.word","adr":int(adr),"width":int(width)})["val"]
    def io_block(self, adrs, width=32):
        return self._send_cmd({"cmd":"io.block","adrs":list(map(int,adrs)),"width":int(width)})["vals"]

    def _io_set_bit(self, adr, val):
        """
        단일 비트 토글 우선 경로: set_bit
        일부 컨트롤러/환경에서 dout(start, '0'/'1')는 'Data length is too much'를 유발할 수 있으므로
        set_bit를 기본으로 사용하고, 미지원일 때만 set로 폴백한다.
        """
        try:
            return self.dio_set_bit(int(adr), 1 if val else 0)
        except Exception as e:
            # 브릿지가 set_bit를 지원하지 않거나 라우팅 실패 시 구 방식으로 폴백
            msg = str(e) or ""
            if "unknown mode" in msg or "set_bit" in msg or "404" in msg:
                return self.dio_set(int(adr), ("1" if val else "0"))
            # i611_io에서 'Data length is too much' 발생 시 비트열 길이 보정 시도
            if "Data length is too much" in msg:
                # 안전하게 1비트만 보낼 수 있도록 보정 (환경에 따라 다를 수 있으나, 우선 단일 비트만 전송)
                return self.dio_set(int(adr), ("1" if val else "0"))
            raise

    def _io_get_bit(self, adr):
        return (self.dio_get(int(adr), 1) != 0)

    # ---------------- Motion Commands (Truncated for brevity - include all from original) ----------------
    def movej(self, joint, profile=None, ik_option=None, **kwargs):
        prof = _apply_profile_and_merge(kwargs, profile)
        cmd = {"cmd":"movej", "joint": list(map(float, joint))}
        cmd.update(prof)
        if ik_option: cmd["ik_option"] = dict(ik_option)
        return self._send_cmd(cmd)

    def movel(self, pose, profile=None, ik_option=None, **kwargs):
        prof = _apply_profile_and_merge(kwargs, profile)
        cmd = {"cmd":"movel", "pose": list(map(float, pose))}
        cmd.update(prof)
        if ik_option: cmd["ik_option"] = dict(ik_option)
        return self._send_cmd(cmd)

    def line(self, pose, profile=None, ik_option=None, **kwargs):
        prof = _apply_profile_and_merge(kwargs, profile)
        cmd = {"cmd":"line", "pose": list(map(float, pose))}
        cmd.update(prof)
        if ik_option: cmd["ik_option"] = dict(ik_option)
        return self._send_cmd(cmd)

    def reljnt(self, delta_j, profile=None, ik_option=None, **kwargs):
        prof = _apply_profile_and_merge(kwargs, profile)
        cmd = {"cmd":"reljnt", "delta_j": list(map(float, delta_j))}
        cmd.update(prof)
        if ik_option: cmd["ik_option"] = dict(ik_option)
        return self._send_cmd(cmd)

    def relline(self, delta_p, profile=None, ik_option=None, **kwargs):
        prof = _apply_profile_and_merge(kwargs, profile)
        cmd = {"cmd":"relline", "delta_p": list(map(float, delta_p))}
        cmd.update(prof)
        if ik_option: cmd["ik_option"] = dict(ik_option)
        return self._send_cmd(cmd)

    # ---------------- Gripper ----------------
    def gripper_close(self, do1=None, do3=None):
        if do1 is None: do1 = self.GRIPPER_DO1
        if do3 is None: do3 = self.GRIPPER_DO3
        self._io_set_bit(do1, 1); time.sleep(0.02)
        self._io_set_bit(do3, 0)
        return {"ok": True}

    def gripper_open(self, do1=None, do3=None):
        if do1 is None: do1 = self.GRIPPER_DO1
        if do3 is None: do3 = self.GRIPPER_DO3
        self._io_set_bit(do1, 0); time.sleep(0.02)
        self._io_set_bit(do3, 1)
        return {"ok": True}

    # ---------------- Pneumatic ----------------
    def pneu_suck(self, require_sv_on=False, wait_sv_on=False, timeout=5.0,
                  suck_do=None, blow_do=None):
        if suck_do is None: suck_do = self.PNEU_SUCK_DO
        if blow_do is None: blow_do = self.PNEU_BLOW_DO
        if require_sv_on or wait_sv_on:
            t0 = time.time()
            while True:
                try:
                    sv_on = 1 if int(self.svstat()[1]) == 1 else 0
                except Exception:
                    sv_on = 0
                if sv_on or not (require_sv_on or wait_sv_on):
                    break
                if time.time() - t0 > float(timeout):
                    raise Timeout("SV ON wait timeout")
                time.sleep(0.05)
            if require_sv_on and (sv_on != 1):
                raise RuntimeError("Servo OFF — pneu_suck blocked by option.")
        self._io_set_bit(blow_do, 0); time.sleep(0.02)
        self._io_set_bit(suck_do, 1)
        return {"ok": True}

    def pneu_blow(self, require_sv_on=False, wait_sv_on=False, timeout=5.0,
                  suck_do=None, blow_do=None):
        if suck_do is None: suck_do = self.PNEU_SUCK_DO
        if blow_do is None: blow_do = self.PNEU_BLOW_DO
        if require_sv_on or wait_sv_on:
            t0 = time.time()
            while True:
                try:
                    sv_on = 1 if int(self.svstat()[1]) == 1 else 0
                except Exception:
                    sv_on = 0
                if sv_on or not (require_sv_on or wait_sv_on):
                    break
                if time.time() - t0 > float(timeout):
                    raise Timeout("SV ON wait timeout")
                time.sleep(0.05)
            if require_sv_on and (sv_on != 1):
                raise RuntimeError("Servo OFF — pneu_blow blocked by option.")
        self._io_set_bit(suck_do, 0); time.sleep(0.02)
        self._io_set_bit(blow_do, 1)
        return {"ok": True}

    # ---------------- Arduino Bridge helpers ----------------
    def arduino_stat(self):
        return self._send_cmd({"cmd":"arduino.stat"})

    def arduino_flush(self):
        return self._send_cmd({"cmd":"arduino.flush"})

    def arduino_exec(self, line, collect_ms=300, timeout=3.0, filter_pong=True):
        r = self._send_cmd({
            "cmd":"arduino.exec",
            "line": str(line),
            "collect_ms": int(collect_ms),
            "timeout": float(timeout)
        })
        lines = [_as_text(x) for x in r.get("lines", [])]
        if filter_pong:
            lines = [x for x in lines if x.strip() != "PONG"]
        return lines

    def arduino_readline(self, timeout=0.0):
        return self._send_cmd({"cmd":"arduino.readline", "timeout": float(timeout)}).get("line")

    # Simple Arduino helpers
    def ar_cm(self, cm): return self.arduino_exec("CM %.3f" % float(cm))
    def ar_speed(self, period_us): return self.arduino_exec("SPEED %d" % int(period_us))
    def ar_ena(self, on=True): return self.arduino_exec("ENA %s" % ("ON" if on else "OFF"))
    def ar_status(self): return self.arduino_exec("STATUS?")
    def ar_ping(self, timeout=1.0):
        try:
            out = self.arduino_exec("PING", collect_ms=200, timeout=timeout, filter_pong=False)
            if any(isinstance(x, basestring) and x.strip() == "PONG" for x in out):
                return True
            st = self.arduino_stat()
            return bool(st.get("connected", False))
        except BridgeError:
            return False
    
    # ---- Arduino STATUS 파서/동기화 ----
    def _parse_status_lines(self, lines):
        """STATUS? 출력에서 CURRENT_POS(mm), STEP_PER(us), STEPS_PER_CM 추출"""
        pos = None; step_per = None; spcm = None
        for t in (lines or []):
            s = (t or "").strip()
            if s.startswith("CURRENT_POS=") and s.endswith("mm"):
                try:
                    pos = float(s.split("CURRENT_POS=")[1].split("mm")[0].strip())
                except Exception:
                    pass
            elif s.startswith("STEP_PER=") and s.endswith("us"):
                try:
                    step_per = int(s.split("STEP_PER=")[1].split("us")[0].strip())
                except Exception:
                    pass
            elif s.startswith("STEPS_PER_CM="):
                try:
                    spcm = float(s.split("STEPS_PER_CM=")[1].strip())
                except Exception:
                    pass
        return pos, step_per, spcm

    def _sync_rail_position_from_arduino(self):
        lines = self.ar_status()
        pos, _, _ = self._parse_status_lines(lines)
        if pos is not None:
            self._rail_position = float(pos)
        return self._rail_position

    def ar_cm_blocking(self, cm, margin_sec=5.0, min_timeout=10.0, max_timeout=3600.0):
        step_period_us, steps_per_cm = self._arduino_params()
        steps = abs(float(cm)) * steps_per_cm
        expected_sec = (steps * max(step_period_us, 1)) / 1e6
        wait_sec = min(max(expected_sec * 1.2 + margin_sec, min_timeout), max_timeout)
        collect_ms = int(wait_sec * 1000.0)

        self.arduino_flush()
        return self.arduino_exec(
            "CM %.3f" % float(cm),
            collect_ms=collect_ms,
            timeout=wait_sec + 5.0,
            filter_pong=True
        )

    def _arduino_params(self):
        step_period_us = 800
        steps_per_cm = 842.2656
        try:
            status_lines = self.ar_status() or []
            for t in status_lines:
                t = (t or "").strip()
                if "STEP_PER=" in t:
                    try:
                        seg = t.split("STEP_PER=")[1]
                        step_period_us = int(seg.split("us")[0].strip())
                    except: pass
                if "STEPS_PER_CM=" in t:
                    try:
                        seg = t.split("STEPS_PER_CM=")[1]
                        steps_per_cm = float(seg.split()[0].strip())
                    except: pass
        except:
            pass
        return step_period_us, steps_per_cm
    
    #-------------------리니어 레일 제어 (확장 기능)---------------------------------#
    def rail_status(self):
        """
        리니어 레일 및 아두이노 연결 상태 확인
        
        Returns:
            dict: {
                "ok": True/False,
                "arduino": {
                    "connected": True/False,
                    "port": "/dev/ttyUSB0",
                    "rx_cached": 0,
                    "last": "..."
                },
                "position": 현재 위치 (mm, 추정값)
            }
        """
        stat = self.arduino_stat()
        # 하드웨어에서 현재 위치를 가져와 반영
        try:
            stat_lines = self.ar_status()
            pos, _, _ = self._parse_status_lines(stat_lines)
            if pos is not None:
                self._rail_position = float(pos)
        except Exception:
            pass
        stat["position"] = float(self._rail_position)
        return stat
    
    def rail_forward(self, distance=None, speed=None, wait=True):
        """
        리니어 레일 전진
        
        Args:
            distance: 이동 거리 (mm), 기본값: RAIL_MAX_STROKE
            speed: 이동 속도 (mm/s), 기본값: RAIL_DEFAULT_SPEED
            wait: 이동 완료까지 대기 여부
        
        Returns:
            dict: {"ok": True/False, "distance": ..., "speed": ..., "lines": [...]}
        """
        if distance is None:
            distance = self.RAIL_MAX_STROKE
        if speed is None:
            speed = self.RAIL_DEFAULT_SPEED
        
        # 상대이동은 하드웨어 좌표 기준으로 처리되므로, 클라이언트 캐시엔 의존하지 않는다.
        
        cmd = "M1 %.2f %.2f" % (distance, speed)
        
        # 예상 시간 계산
        expected_time = abs(distance) / speed + 1.0
        timeout = expected_time + 3.0
        
        try:
            lines = self.arduino_exec(cmd, collect_ms=int(expected_time * 1000), timeout=timeout)
            if wait:
                self._wait_until_idle(expected_time)
            # 완료 후 실제 위치를 읽어 캐시 갱신
            self._sync_rail_position_from_arduino()
            
            return {
                "ok": True,
                "distance": distance,
                "speed": speed,
                "direction": "forward",
                "lines": lines,
                "current_position": self._rail_position
            }
        except Exception as e:
            return {"ok": False, "err": str(e)}
    
    def rail_backward(self, distance=None, speed=None, wait=True):
        """
        리니어 레일 후진
        
        Args:
            distance: 이동 거리 (mm), 기본값: RAIL_MAX_STROKE
            speed: 이동 속도 (mm/s), 기본값: RAIL_DEFAULT_SPEED
            wait: 이동 완료까지 대기 여부
        
        Returns:
            dict: {"ok": True/False, "distance": ..., "speed": ..., "lines": [...]}
        """
        if distance is None:
            distance = self.RAIL_MAX_STROKE
        if speed is None:
            speed = self.RAIL_DEFAULT_SPEED
        
        # 상대이동은 하드웨어 좌표 기준으로 처리
        
        cmd = "M1 -%.2f %.2f" % (distance, speed)
        
        # 예상 시간 계산
        expected_time = abs(distance) / speed + 1.0
        timeout = expected_time + 3.0
        
        try:
            lines = self.arduino_exec(cmd, collect_ms=int(expected_time * 1000), timeout=timeout)
            if wait:
                self._wait_until_idle(expected_time)
            self._sync_rail_position_from_arduino()
            
            return {
                "ok": True,
                "distance": distance,
                "speed": speed,
                "direction": "backward",
                "lines": lines,
                "current_position": self._rail_position
            }
        except Exception as e:
            return {"ok": False, "err": str(e)}
    
    def rail_home(self, speed=None, wait=True):
        """
        리니어 레일 원점 복귀 (0mm 위치로 이동)
        
        Args:
            speed: 이동 속도 (mm/s), 기본값: RAIL_HOME_SPEED
            wait: 이동 완료까지 대기 여부
        
        Returns:
            dict: {"ok": True/False, ...}
        """
        if speed is None:
            speed = self.RAIL_HOME_SPEED
        
        try:
            lines = self.arduino_exec("HOME", collect_ms=800, timeout=12.0)
            if wait:
                # 폴링으로 0mm 근처까지 대기
                self._wait_until_position(0.0, tol_mm=1.0, max_wait=20.0)
            self._sync_rail_position_from_arduino()
            
            return {
                "ok": True,
                "lines": lines,
                "current_position": self._rail_position
            }
        except Exception as e:
            return {"ok": False, "err": str(e)}
    
    def rail_move_to(self, position, speed=None, wait=True):
        """
        리니어 레일을 절대 위치로 이동
        
        Args:
            position: 목표 위치 (mm, 0 ~ RAIL_MAX_STROKE)
            speed: 이동 속도 (mm/s), 기본값: RAIL_DEFAULT_SPEED
            wait: 이동 완료까지 대기 여부
        
        Returns:
            dict: {"ok": True/False, ...}
        """
        if speed is None:
            speed = self.RAIL_DEFAULT_SPEED
        
        # 범위 체크
        if position < 0:
            position = 0
        elif position > self.RAIL_MAX_STROKE:
            position = self.RAIL_MAX_STROKE
        
        cmd = "M2 %.2f %.2f" % (position, speed)
        
        distance = abs(position - self._rail_position)
        expected_time = distance / speed + 1.0
        timeout = expected_time + 3.0
        
        try:
            lines = self.arduino_exec(cmd, collect_ms=int(expected_time * 1000), timeout=timeout)
            if wait:
                self._wait_until_position(position, tol_mm=1.0, max_wait=max(10.0, expected_time+5.0))
            self._sync_rail_position_from_arduino()
            
            return {
                "ok": True,
                "target_position": position,
                "speed": speed,
                "lines": lines,
                "current_position": self._rail_position
            }
        except Exception as e:
            return {"ok": False, "err": str(e)}
    
    def rail_set_position(self, position):
        """
        현재 위치를 지정된 값으로 설정 (캘리브레이션용)
        
        Args:
            position: 설정할 위치 값 (mm)
        
        Returns:
            dict: {"ok": True, "position": ...}
        """
        self._rail_position = float(position)
        return {"ok": True, "position": self._rail_position}
    
    def rail_get_position(self):
        """
        현재 레일 위치 반환 (하드웨어에서 동기화)
        """
        try:
            return float(self._sync_rail_position_from_arduino())
        except Exception:
            return float(self._rail_position)

    # ---- 내부: 위치/아이들 대기 ----
    def _wait_until_idle(self, expected_time, extra=1.0, max_wait=30.0):
        """간단한 시간 대기 + STATUS 폴링 한 번 (안정화)"""
        t_wait = min(max(expected_time + extra, 0.0), max_wait)
        if t_wait > 0:
            time.sleep(t_wait)
        try:
            self._sync_rail_position_from_arduino()
        except Exception:
            pass

    def _wait_until_position(self, target_mm, tol_mm=1.0, max_wait=20.0, poll=0.2):
        """STATUS?를 폴링해 목표 위치 근처까지 대기"""
        t0 = time.time()
        last = None
        while time.time() - t0 < max_wait:
            try:
                pos = self._sync_rail_position_from_arduino()
                last = pos
                if abs(float(pos) - float(target_mm)) <= float(tol_mm):
                    return True
            except Exception:
                pass
            time.sleep(poll)
        # 마지막 위치를 캐시에 남겨두고 종료
        if last is not None:
            self._rail_position = float(last)
        return False
    
    # 기존 액추에이터 호환 메서드 (rail_* 메서드를 사용하는 것을 권장)
    def actuator_status(self):
        """아두이노 연결 상태 확인 (rail_status()와 동일)"""
        return self.rail_status()
    
    def actuator_forward(self, distance=800, speed=300):
        """리니어 레일 전진 (rail_forward()와 동일)"""
        return self.rail_forward(distance, speed, wait=True)
    
    def actuator_backward(self, distance=800, speed=300):
        """리니어 레일 후진 (rail_backward()와 동일)"""
        return self.rail_backward(distance, speed, wait=True)
    
    def actuator_move(self, command):
        """
        간단한 명령어로 액추에이터 제어
        Args:
            command: 1=전진, 2=후진, 0=홈
        """
        if command == 1:
            return self.rail_forward()
        elif command == 2:
            return self.rail_backward()
        elif command == 0:
            return self.rail_home()
        else:
            raise ValueError("Invalid command. Use 0 (home), 1 (forward) or 2 (backward)")

    # ---------------- Convenience moves ----------------
    def _chunked_relline(self, dx, dy, dz, drz=0, dry=0, drx=0, step_mm=None, aprof=None, ik_option=None):
        if step_mm is None: step_mm = self.LIN_STEP_MM
        if aprof   is None: aprof   = self.DEFAULT_PROFILE
        def _seq(total, s):
            n = int(abs(total) // s); rem = abs(total) - n*s
            out = [copysign(s, total)] * n
            if rem > 1e-6: out.append(copysign(rem, total))
            return out
        xs, ys, zs = _seq(dx, step_mm), _seq(dy, step_mm), _seq(dz, step_mm)
        rzseq, ryseq, rxseq = _seq(drz, self.ANG_STEP_DEG), _seq(dry, self.ANG_STEP_DEG), _seq(drx, self.ANG_STEP_DEG)
        L = max(len(xs), len(ys), len(zs), len(rzseq), len(ryseq), len(rxseq))
        for i in range(L):
            ddx = xs[i] if i < len(xs) else 0.0
            ddy = ys[i] if i < len(ys) else 0.0
            ddz = zs[i] if i < len(zs) else 0.0
            rrz = rzseq[i] if i < len(rzseq) else 0.0
            rry = ryseq[i] if i < len(ryseq) else 0.0
            rrx = rxseq[i] if i < len(rxseq) else 0.0
            self.relline([ddx, ddy, ddz, rrz, rry, rrx], profile=aprof, ik_option=ik_option)

    def _chunked_reljnt(self, dj, step_deg=None, aprof=None, ik_option=None):
        if step_deg is None: step_deg = self.RELJ_STEP_DEG
        if aprof    is None: aprof    = self.DEFAULT_PROFILE
        parts = []
        for v in dj:
            n = int(abs(v) // step_deg); rem = abs(v) - n*step_deg
            seq = [copysign(step_deg, v)] * n
            if rem > 1e-6: seq.append(copysign(rem, v))
            parts.append(seq)
        L = max(map(len, parts)) if parts else 0
        for i in range(L):
            inc = [parts[j][i] if i < len(parts[j]) else 0.0 for j in range(6)]
            self.reljnt(inc, profile=aprof, ik_option=ik_option)

    def go_home(self, confirm=False, smooth=True, profile=None, ik_option=None):
        if profile is None: profile = self.DEFAULT_PROFILE
        target = [0.0]*6
        return self.movej(target, profile=profile, ik_option=ik_option)

    def save_zero_encoder_here(self):
        _, curr_joint, *_ = self.where_conv()
        self.ZERO_ENCODER_JOINT = list(curr_joint)
        return {"ok": True, "zero_encoder_joint": self.ZERO_ENCODER_JOINT}

    def go_zero_encoder(self, confirm=False, profile=None, use_saved=True, ik_option=None):
        if profile is None: profile = self.DEFAULT_PROFILE
        target = self.ZERO_ENCODER_JOINT if use_saved else [-0.01, 0.0, 180.0, -0.02, 0.14, -0.03]
        return self.movej(target, profile=profile, ik_option=ik_option)

    def align_hand(self, target_j6=0.0, profile=None, ik_option=None):
        if profile is None: profile = self.DEFAULT_PROFILE
        _, curr_joint, *_ = self.where_conv()
        j = list(curr_joint); j[5] = float(target_j6)
        return self.movej(j, profile=profile, ik_option=ik_option)


if __name__ == '__main__':
    host = "127.0.0.1"
    try:
        import sys
        if len(sys.argv) > 1:
            host = sys.argv[1]
    except Exception:
        pass

    with BridgeClient(host=host) as bc:
        print('===== Robot Bridge Test =====')
        print('HELLO', bc.hello())
        print('SV', bc.svstat())
        
        print('\n===== IK Option Test =====')
        print('IK OPT (init)', bc.ik_get_option())
        print('SET IK', bc.ik_set_option({"prefer": "current_elbow", "flip_ok": False}))
        print('GET IK', bc.ik_get_option())
        
        print('\n===== Linear Rail Test =====')
        print('Rail Status:', bc.rail_status())
        print('Arduino Ping:', bc.ar_ping())
        
        # 테스트용 이동 (주의: 실제 하드웨어 연결 시에만 실행)
        # print('Rail Forward 100mm:', bc.rail_forward(100, 200))
        # print('Rail Position:', bc.rail_get_position())
        # print('Rail Home:', bc.rail_home())
