/*
 리니어 액추에이터 통합 제어 (Robot Bridge v1.3 호환)
 * 하드웨어 연결:
 * [스테퍼 모터 드라이버]
 * PUL+ → Pin 2
 * DIR+ → Pin 3
 * ENA+ → Pin 4
 * PUL-, DIR-, ENA- → GND
 * 시리얼 통신: 115200bps (브릿지 서버와 통신)
*/

// ===== 함수 프로토타입 =====
void printMenu();
void handleMotorCommand(String cmd);
void motorMoveRelative(float distanceMm, float speedMmPerSec);
void motorMoveToPosition(float targetMm, float speedMmPerSec);
void sendStatus();

// 모터 제어 핀
#define PUL_PIN 2  // PUL+ 연결 핀
#define DIR_PIN 3  // DIR+ 연결 핀
#define ENA_PIN 4  // EN+ 연결 핀
// 주의: PUL-, DIR-, EN- 는 모두 Arduino GND에 연결!

// ==================== 모터 파라미터 ====================
const float MAX_STROKE = 800.0;          // 최대 이동 거리 (mm)
float currentPosition = 0.0;             // 현재 위치 (mm)
const int   PULSES_PER_REV = 400;        // 드라이버 설정: 1회전당 펄스 수 (400)
const float MM_PER_REV     = 59.0;       // 1회전당 이동 거리 (mm) - 보정됨
const float STEPS_PER_MM   = PULSES_PER_REV / MM_PER_REV;  // mm당 필요한 펄스 수

// 가속/감속 파라미터
const int ACCEL_STEPS = 200;             // 가속/감속 구간 펄스 수
const int MIN_DELAY   = 100;             // 최소 딜레이 (us) - 최대 속도
const int MAX_DELAY   = 800;             // 최대 딜레이 (us) - 시작/종료 속도

// ==================== 초기화 ====================
void setup() {
  Serial.begin(115200);  // 브릿지 서버와 통신하기 위해 115200으로 설정
  while (!Serial);
  
  // ID 전송 (브릿지 서버가 인식용)
  Serial.println("ID:TB6600CTL");
  Serial.println("Arduino Linear Rail Controller v1.3");

  delay(100);

  pinMode(PUL_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENA_PIN, OUTPUT);

  digitalWrite(PUL_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);
  digitalWrite(ENA_PIN, LOW);  // LOW = 모터 활성화(드라이버에 따라 반대일 수 있음)

  Serial.println("✓ 모터 핀 초기화 완료");
  Serial.print("  최대 스트로크: "); Serial.print(MAX_STROKE); Serial.println("mm");
  Serial.print("  펄스/회전: ");     Serial.println(PULSES_PER_REV);
  Serial.print("  mm/회전: ");       Serial.println(MM_PER_REV);
  Serial.print("  펄스/mm: ");       Serial.println(STEPS_PER_MM, 2);

  printMenu(); // 프로그램 시작 시 메뉴 호출
}

// ==================== 메인 루프 ====================
void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();
    
    if (msg.length() == 0) return;

    // ID 질의(브릿지 장치 식별용) 처리 - 브릿지가 포트 오픈 직후 보낼 수 있으니 가장 먼저 처리
    if (msg == "ID?" || msg == "ID") {
      Serial.println("ID:TB6600CTL");
      return;
    }

    // PING 명령 처리 (브릿지 서버 헬스체크용)
    if (msg == "PING" || msg == "ping") {
      Serial.println("PONG");
      return;
    }
    
    // STATUS 명령 처리
    if (msg == "STATUS?" || msg == "STATUS") {
      sendStatus();
      return;
    }
    
    // HOME 명령 처리 (원점 복귀)
    if (msg == "HOME") {
      Serial.println("→ 원점 복귀 시작");
      motorMoveToPosition(0.0, 200.0);
      return;
    }

    // 모터 명령 처리
    if (msg.startsWith("M")) {
      handleMotorCommand(msg);
    } else {
      Serial.println("✗ 알 수 없는 명령어");
    }
  }
}

// ==================== 상태 정보 전송 ====================
void sendStatus() {
  Serial.print("STEP_PER=");
  Serial.print((MIN_DELAY + MAX_DELAY) / 2);
  Serial.println("us");
  
  Serial.print("STEPS_PER_CM=");
  Serial.println(STEPS_PER_MM * 10.0, 2);
  
  Serial.print("CURRENT_POS=");
  Serial.print(currentPosition, 2);
  Serial.println("mm");
  
  Serial.print("MAX_STROKE=");
  Serial.print(MAX_STROKE);
  Serial.println("mm");
}

// ==================== 메뉴 출력 ====================
void printMenu() {
  Serial.println("\n===== [모터 제어] =====");
  Serial.println("M1 <거리(mm)> [속도(mm/s)] - 상대 이동 (예: M1 50 120)");
  Serial.println("M2 <위치(mm)> [속도(mm/s)] - 절대 이동 (예: M2 400 200)");
  Serial.println("HOME - 원점 복귀");
  Serial.println("JT <delay_us> <pulses> - 지터 테스트(고정 딜레이 펄스 생성)");
  Serial.println("STATUS? - 상태 확인");
  Serial.println("PING - 연결 확인");
  Serial.print("명령어 입력 > ");
}

// ==================== 모터 제어 함수 ====================
void handleMotorCommand(String cmd) {
  cmd.trim();

  int firstSpace = cmd.indexOf(' ');
  String motorCmd = (firstSpace == -1) ? cmd : cmd.substring(0, firstSpace);
  String params   = (firstSpace == -1) ? ""  : cmd.substring(firstSpace + 1);
  params.trim();

  if (motorCmd == "JT") {
    // JT <delay_us> <pulses>
    long dly = 200; long pls = 2000;
    if (params.length() > 0) {
      int sp = params.indexOf(' ');
      if (sp == -1) { dly = params.toInt(); }
      else { dly = params.substring(0, sp).toInt(); pls = params.substring(sp + 1).toInt(); }
    }
    if (dly < 2) dly = 2;
    if (pls < 10) pls = 10;
    Serial.print("지터 테스트: delay_us="); Serial.print(dly); Serial.print(", pulses="); Serial.println(pls);
    unsigned long tmin=1000000000UL, tmax=0, tsum=0;
    pinMode(PUL_PIN, OUTPUT);
    unsigned long t0 = micros(), t_prev = micros();
    for (long i=0;i<pls;i++){
      digitalWrite(PUL_PIN, HIGH);
      delayMicroseconds(dly);
      digitalWrite(PUL_PIN, LOW);
      delayMicroseconds(dly);
      unsigned long t_now = micros();
      unsigned long dt = t_now - t_prev; t_prev = t_now;
      if (dt < tmin) tmin = dt; if (dt > tmax) tmax = dt; tsum += dt;
      // 중간에 PING 오면 응답
      if ((i & 0xFF) == 0 && Serial.available()){
        String s = Serial.readStringUntil('\n'); s.trim();
        if (s == "PING" || s == "ping") Serial.println("PONG");
      }
    }
    unsigned long total = micros() - t0;
    Serial.print("완료. 총시간(us)="); Serial.print(total);
    Serial.print(", 주기(us) min/avg/max ≈ ");
    Serial.print(tmin); Serial.print("/");
    Serial.print(tsum / (unsigned long)pls); Serial.print("/");
    Serial.println(tmax);
    Serial.print("명령어 입력 > ");
    return;
  }

  if (motorCmd == "M1") {
    // 상대 이동: M1 <거리(mm)> [속도(mm/s)]
    if (params.length() == 0) {
      Serial.println("✗ 형식: M1 <거리(mm)> [속도(mm/s)]");
      return;
    }

    // 첫 번째 값 (거리)
    int sp = params.indexOf(' ');
    float distance = 0.0f;
    float speed    = 200.0f; // 기본 속도

    if (sp == -1) {
      distance = params.toFloat();
    } else {
      String s1 = params.substring(0, sp); s1.trim();
      String s2 = params.substring(sp + 1); s2.trim();
      distance = s1.toFloat();
      if (s2.length() > 0) speed = s2.toFloat();
    }

    motorMoveRelative(distance, speed);
  }
  else if (motorCmd == "M2") {
    // 절대 위치 이동: M2 <위치(mm)> [속도(mm/s)]
    if (params.length() == 0) {
      Serial.println("✗ 형식: M2 <위치(mm)> [속도(mm/s)]");
      return;
    }

    int sp = params.indexOf(' ');
    float target = 0.0f;
    float speed  = 200.0f;

    if (sp == -1) {
      target = params.toFloat();
    } else {
      String s1 = params.substring(0, sp); s1.trim();
      String s2 = params.substring(sp + 1); s2.trim();
      target = s1.toFloat();
      if (s2.length() > 0) speed = s2.toFloat();
    }

    motorMoveToPosition(target, speed);
  }
  else {
    Serial.println("✗ 알 수 없는 모터 명령어");
  }
}

// ==================== 이동 구현 ====================
void motorMoveRelative(float distanceMm, float speedMmPerSec) {
  float targetPosition = currentPosition + distanceMm;

  // 범위 체크 및 보정
  if (targetPosition < 0) {
    Serial.println("⚠ 목표 위치가 0보다 작습니다. 0으로 이동합니다.");
    targetPosition = 0;
    distanceMm = targetPosition - currentPosition;
  } else if (targetPosition > MAX_STROKE) {
    Serial.print("⚠ 목표 위치가 최대값을 초과합니다. ");
    Serial.print(MAX_STROKE); Serial.println("mm로 이동합니다.");
    targetPosition = MAX_STROKE;
    distanceMm = targetPosition - currentPosition;
  }

  if (abs(distanceMm) < 0.01) {
    Serial.println("이동 거리가 0입니다.");
    return;
  }

  // 방향 설정
  bool forward = distanceMm > 0;
  digitalWrite(DIR_PIN, forward ? HIGH : LOW);

  Serial.print("→ ");
  Serial.print(forward ? "전진" : "후진");
  Serial.print(" ");
  Serial.print(abs(distanceMm), 2);
  Serial.print("mm (속도: ");
  Serial.print(speedMmPerSec);
  Serial.println("mm/s)");

  // 펄스 수 계산
  long totalPulses = (long)(abs(distanceMm) * STEPS_PER_MM);

  // 디버깅 정보 출력
  Serial.print("  필요한 펄스: "); Serial.println(totalPulses);
  Serial.print("  현재→목표: ");
  Serial.print(currentPosition, 2);
  Serial.print("mm → ");
  Serial.print(targetPosition, 2);
  Serial.println("mm");

  // 목표 딜레이 계산 (us)
  unsigned long targetDelay = (unsigned long)(500000.0 / (speedMmPerSec * STEPS_PER_MM));

  if (targetDelay < MIN_DELAY) {
    targetDelay = MIN_DELAY;
    Serial.println("  ⚠ 속도 제한 적용");
  }
  if (targetDelay > MAX_DELAY) {
    targetDelay = MAX_DELAY;
  }

  // 가속/등속/감속
  const long SERIAL_POLL_EVERY = 200;  // N펄스마다 시리얼 체크
  for (long i = 0; i < totalPulses; i++) {
    unsigned long currentDelay;

    if (i < ACCEL_STEPS) {
      // 가속: MAX_DELAY → targetDelay
      currentDelay = MAX_DELAY - ((MAX_DELAY - targetDelay) * i / ACCEL_STEPS);
    } else if (i > totalPulses - ACCEL_STEPS) {
      // 감속: targetDelay → MAX_DELAY
      long remainingPulses = totalPulses - i;
      currentDelay = targetDelay + ((MAX_DELAY - targetDelay) * (ACCEL_STEPS - remainingPulses) / ACCEL_STEPS);
    } else {
      // 등속
      currentDelay = targetDelay;
    }

    digitalWrite(PUL_PIN, HIGH);
    delayMicroseconds(currentDelay);
    digitalWrite(PUL_PIN, LOW);
    delayMicroseconds(currentDelay);

    // --- 모션 중 경량 시리얼 핑 처리(브릿지 PING 대응) ---
    if (i % SERIAL_POLL_EVERY == 0) {
      if (Serial.available()) {
        String _tmp = Serial.readStringUntil('\n');
        _tmp.trim();
        if (_tmp == "PING" || _tmp == "ping") {
          Serial.println("PONG");
        } else if (_tmp == "ID?" || _tmp == "ID") {
          Serial.println("ID:TB6600CTL");
        }
        // 그 외 명령은 모션 중 무시
      }
    }

    // 진행 상황 표시 (10% 단위)
    if (totalPulses > 100 && i > 0 && totalPulses / 10 > 0 && i % (totalPulses / 10) == 0) {
      Serial.print(".");
    }
  }

  currentPosition = targetPosition;
  Serial.println("\n✓ 이동 완료");
  Serial.print("  현재 위치: ");
  Serial.print(currentPosition, 2);
  Serial.println("mm");
}

void motorMoveToPosition(float targetMm, float speedMmPerSec) {
  Serial.print("목표 위치: ");
  Serial.print(targetMm, 2);
  Serial.println("mm");

  float distance = targetMm - currentPosition;
  motorMoveRelative(distance, speedMmPerSec);
}
