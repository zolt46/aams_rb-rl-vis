from robot_sdk_v13_rail_extended import BridgeClient

bc = BridgeClient(host="192.168.1.23")
bc.connect()
print("HELLO:", bc.hello())

# (A) 아두이노 연결/상태
print("Arduino stat:", bc.rail_status())
print("Arduino ping:", bc.ar_ping())
print("Arduino STATUS?:", bc.ar_status())

# (B) 지터(펄스 주기) 테스트: delay_us=200, pulses=4000
#    결과에서 min/avg/max 범위가 좁을수록 좋습니다.
print("JITTER:", bc.arduino_exec("JT 200 4000", collect_ms=2000, timeout=5))

# (C) 저속 → 중속 → 고속 순으로 50mm씩 전진해보며 떨림 관찰
for v in [100, 150, 200, 250, 300]:  # mm/s
    print(f"\n>>> forward 50mm @ {v}mm/s")
    r = bc.rail_forward(distance=50, speed=v, wait=True)
    print(r)

# (D) 홈 복귀
print("\n>>> HOME")
print(bc.rail_home(speed=200, wait=True))

bc.close()