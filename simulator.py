"""
simulator.py
───────────────
데이터센터 냉각 물리 시뮬레이터.

[모델 구조]
실제 데이터센터를 단순화한 열역학 모델:
  - 서버 랙 N개: 각자 CPU 부하에 따라 발열
  - 냉각 시스템: CRAC(Computer Room Air Conditioner) 유닛
  - 외기: 외부 온도가 냉각 효율에 영향

[열 전달 방정식]
  dT_room/dt = (Q_server - Q_cooling) / (m_air * c_air)

  Q_server  : 서버 총 발열량 [kW]
  Q_cooling : 냉각 시스템 제거 열량 [kW]
  m_air     : 실내 공기 질량 [kg]
  c_air     : 공기 비열 [kJ/kg·K]

[PUE (Power Usage Effectiveness)]
  PUE = 총 시설 전력 / IT 장비 전력
  PUE = 1.0 이 이상적, 실제 평균 1.5~2.0
  DeepMind 적용 후 Google 데이터센터 PUE ≈ 1.12
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerRack:
    """서버 랙 1개의 열 모델."""
    rack_id:      int
    max_power_kw: float = 10.0    # 최대 전력 소비 [kW]
    base_load:    float = 0.3     # 기본 부하율 (항상 켜진 프로세스)

    # 상태 (시뮬레이션 중 변경)
    cpu_load:     float = 0.5     # 현재 CPU 부하 [0, 1]
    temp_c:       float = 25.0    # 랙 내부 온도 [°C]

    @property
    def power_kw(self) -> float:
        """현재 전력 소비 [kW]. 부하에 비례."""
        return self.max_power_kw * (self.base_load + (1 - self.base_load) * self.cpu_load)

    @property
    def heat_kw(self) -> float:
        """발열량 = 전력 소비 거의 전부가 열로 변환."""
        return self.power_kw * 0.97


@dataclass
class CoolingUnit:
    """
    CRAC(Computer Room Air Conditioner) 유닛.
    냉각 팬 속도와 냉수 유량 두 가지 제어 변수.
    """
    unit_id:       int
    max_cool_kw:   float = 50.0   # 최대 냉각 능력 [kW]

    # 제어 변수 (에이전트가 결정)
    fan_speed:     float = 0.5    # 팬 속도 [0, 1]
    chilled_water_flow: float = 0.5   # 냉수 유량 [0, 1]

    @property
    def cooling_power_kw(self) -> float:
        """현재 냉각 전력 소비 [kW]. 팬(fan law) + 냉수 칠러 압축기.
        팬만 계산하면 PUE 변동 폭이 0.02 수준으로 너무 좁아 에이전트가
        에너지 최적화 신호를 거의 받지 못함. 칠러 항을 포함해야 PUE 1.1~1.8
        범위가 만들어져 실질적인 최적화가 가능해진다."""
        fan_power     = self.max_cool_kw * 0.10 * (self.fan_speed ** 3 + 0.05)
        chiller_power = self.max_cool_kw * 0.25 * self.chilled_water_flow
        return fan_power + chiller_power

    @property
    def cooling_capacity_kw(self) -> float:
        """실제 냉각 용량: 팬 + 냉수 유량의 복합 함수."""
        fan_contrib   = self.fan_speed * 0.6
        water_contrib = self.chilled_water_flow * 0.4
        return self.max_cool_kw * (fan_contrib + water_contrib)


class DataCenterSimulator:
    """
    데이터센터 열 역학 시뮬레이터.

    시뮬레이션 단위: 5분 간격 (dt = 300초)
    에피소드 길이  : 24시간 = 288 step
    """

    def __init__(
        self,
        n_racks:       int   = 10,
        n_cooling:     int   = 3,
        dt_sec:        float = 300.0,    # 5분 간격
        t_target_c:    float = 22.0,     # 목표 온도 [°C]
        t_max_c:       float = 35.0,     # 안전 상한 온도 [°C]
        t_min_c:       float = 18.0,     # 결로 방지 하한 [°C]
        seed:          Optional[int] = 42,
    ):
        self.n_racks    = n_racks
        self.n_cooling  = n_cooling
        self.dt         = dt_sec
        self.t_target   = t_target_c
        self.t_max      = t_max_c
        self.t_min      = t_min_c
        self.rng        = np.random.default_rng(seed)

        # 물리 상수
        self.air_mass_kg   = 5000.0    # 실내 공기 질량 [kg]
        self.c_air         = 1.005     # 공기 비열 [kJ/kg·K]
        self.thermal_inertia = self.air_mass_kg * self.c_air   # [kJ/K]

        # 컴포넌트 초기화
        self.racks   = [ServerRack(i, max_power_kw=8.0 + 4.0 * self.rng.random())
                        for i in range(n_racks)]
        self.coolers = [CoolingUnit(i) for i in range(n_cooling)]

        # 환경 상태
        self.room_temp_c   = 24.0
        self.outdoor_temp_c = 15.0
        self.step_count    = 0
        self.scenario      = "normal"          # 현재 에피소드 부하 시나리오
        self._load_profile = self._generate_load_profile(self.scenario)

    # ──────────────────────────────────────────
    # 부하 시나리오 (에피소드별 랜덤 선택)
    # ──────────────────────────────────────────
    LOAD_SCENARIOS = ("normal", "high", "low", "variable")

    def _generate_load_profile(self, scenario: str = "normal") -> np.ndarray:
        """
        부하 시나리오별로 24시간(288 step) CPU 부하 프로파일 생성.

        - normal  : 평일 업무 패턴 — 주간 피크 (9시·14시)
        - high    : 고부하 — AI 학습·배치 작업 집중 (전반적으로 높음)
        - low     : 저부하 — 야간·주말 (낮고 변동 작음)
        - variable: 불규칙 — 기본 패턴 + 랜덤 스파이크 1~3회
        """
        steps = 288
        t     = np.linspace(0, 2 * np.pi, steps)

        if scenario == "high":
            # 고부하: 높은 기저부하 + 강한 주간 피크
            base    = 0.75
            profile = base + 0.18 * np.sin(t - np.pi / 4) \
                           + 0.07 * np.sin(2 * t) \
                           + self.rng.normal(0, 0.05, steps)

        elif scenario == "low":
            # 저부하: 낮은 기저부하 + 완만한 변동
            base    = 0.33
            profile = base + 0.12 * np.sin(t - np.pi / 4) \
                           + self.rng.normal(0, 0.03, steps)

        elif scenario == "variable":
            # 불규칙: 기본 패턴 위에 가우시안 스파이크 1~3회를 무작위 삽입
            base    = 0.50
            profile = base + 0.18 * np.sin(t - np.pi / 4) \
                           + self.rng.normal(0, 0.06, steps)
            n_spikes = int(self.rng.integers(1, 4))         # 1~3회
            idx      = np.arange(steps)
            for _ in range(n_spikes):
                center = int(self.rng.integers(0, steps))
                width  = int(self.rng.integers(6, 18))      # 30분~1.5시간
                amp    = 0.30 + 0.25 * self.rng.random()
                profile = profile + amp * np.exp(-0.5 * ((idx - center) / width) ** 2)

        else:  # "normal" (기본값)
            base    = 0.55
            profile = base + 0.25 * np.sin(t - np.pi / 4) \
                           + 0.10 * np.sin(2 * t) \
                           + self.rng.normal(0, 0.04, steps)

        return np.clip(profile, 0.2, 0.98)

    def _outdoor_temp(self) -> float:
        """외기 온도: 일교차 시뮬레이션."""
        hour   = (self.step_count * 5 / 60) % 24
        base   = 15.0
        swing  = 8.0 * np.sin((hour - 6) * np.pi / 12)
        noise  = self.rng.normal(0, 0.5)
        return base + swing + noise

    # ──────────────────────────────────────────
    # 핵심: 열 역학 업데이트
    # ──────────────────────────────────────────
    def step_physics(self, fan_speeds: np.ndarray,
                     water_flows: np.ndarray) -> dict:
        """
        냉각 제어 입력을 받아 물리 상태를 한 step 진행.

        Args:
            fan_speeds  : 냉각 유닛별 팬 속도 [0,1] 배열
            water_flows : 냉각 유닛별 냉수 유량 [0,1] 배열

        Returns:
            dict: 업데이트된 상태 정보
        """
        # 부하 업데이트
        load = self._load_profile[self.step_count % len(self._load_profile)]
        for rack in self.racks:
            rack.cpu_load = float(np.clip(
                load + self.rng.normal(0, 0.05), 0.1, 1.0
            ))

        # 냉각 유닛 제어 적용
        for i, cooler in enumerate(self.coolers):
            cooler.fan_speed          = float(np.clip(fan_speeds[i],   0, 1))
            cooler.chilled_water_flow = float(np.clip(water_flows[i],  0, 1))

        # 발열 합계
        total_heat_kw = sum(r.heat_kw for r in self.racks)

        # 냉각 용량 합계
        total_cooling_kw = sum(c.cooling_capacity_kw for c in self.coolers)

        # 외기 영향 (자연 열교환)
        self.outdoor_temp_c = self._outdoor_temp()
        natural_exchange_kw = (
            (self.outdoor_temp_c - self.room_temp_c) * 0.5
        )

        # 열 수지로 실내 온도 업데이트
        net_heat_kw = total_heat_kw - total_cooling_kw + natural_exchange_kw
        delta_t     = (net_heat_kw * self.dt) / (self.thermal_inertia)
        self.room_temp_c = float(np.clip(
            self.room_temp_c + delta_t, 10.0, 50.0
        ))

        # 각 랙 온도 업데이트 (실내 온도 + 랙 자체 발열)
        for rack in self.racks:
            rack_delta = (rack.heat_kw * 0.1 * self.dt) / (self.thermal_inertia / self.n_racks)
            rack.temp_c = float(np.clip(
                self.room_temp_c + rack_delta + self.rng.normal(0, 0.3),
                self.room_temp_c, self.room_temp_c + 15
            ))

        # 전력 집계
        it_power_kw      = sum(r.power_kw for r in self.racks)
        cooling_power_kw = sum(c.cooling_power_kw for c in self.coolers)
        total_power_kw   = it_power_kw + cooling_power_kw

        # PUE
        pue = total_power_kw / max(it_power_kw, 0.1)

        self.step_count += 1

        return {
            "room_temp_c":       self.room_temp_c,
            "outdoor_temp_c":    self.outdoor_temp_c,
            "rack_temps":        [r.temp_c for r in self.racks],
            "cpu_loads":         [r.cpu_load for r in self.racks],
            "it_power_kw":       it_power_kw,
            "cooling_power_kw":  cooling_power_kw,
            "total_power_kw":    total_power_kw,
            "pue":               pue,
            "total_heat_kw":     total_heat_kw,
            "total_cooling_kw":  total_cooling_kw,
            "fan_speeds":        fan_speeds.tolist(),
            "water_flows":       water_flows.tolist(),
        }

    def reset(self, seed=None, scenario: Optional[str] = None):
        """시뮬레이터 초기화.

        scenario를 지정하지 않으면 4종(normal/high/low/variable) 중
        에피소드마다 무작위로 선택한다. seed를 주면 시나리오 선택과
        부하 프로파일이 모두 재현 가능하다.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.scenario = (
            scenario if scenario in self.LOAD_SCENARIOS
            else self.LOAD_SCENARIOS[int(self.rng.integers(0, len(self.LOAD_SCENARIOS)))]
        )
        self.room_temp_c  = 22.0 + self.rng.normal(0, 1)
        self.step_count   = 0
        self._load_profile = self._generate_load_profile(self.scenario)
        for rack in self.racks:
            rack.cpu_load = 0.5
            rack.temp_c   = self.room_temp_c + 2.0
        for cooler in self.coolers:
            cooler.fan_speed          = 0.5
            cooler.chilled_water_flow = 0.5


# ──────────────────────────────────────────────
# 동작 확인
# ──────────────────────────────────────────────
if __name__ == "__main__":
    sim = DataCenterSimulator(n_racks=10, n_cooling=3)
    sim.reset()

    print("시뮬레이터 동작 확인 (10 step)")
    print(f"{'Step':>4} | {'실내온도':>7} | {'IT전력':>7} | {'냉각전력':>8} | {'PUE':>5}")
    print("-" * 50)

    for step in range(10):
        # 고정 제어 입력 (팬 50%, 냉수 50%)
        fans   = np.full(3, 0.5)
        waters = np.full(3, 0.5)
        info   = sim.step_physics(fans, waters)
        print(
            f"{step+1:>4} | "
            f"{info['room_temp_c']:>6.2f}°C | "
            f"{info['it_power_kw']:>6.1f}kW | "
            f"{info['cooling_power_kw']:>7.1f}kW | "
            f"{info['pue']:>5.3f}"
        )
