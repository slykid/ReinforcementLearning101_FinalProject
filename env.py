"""
env.py
─────────
데이터센터 냉각 제어 Gymnasium 환경.

[MDP 설계]

State (연속, 정규화된 값):
  - 실내 온도          (1)
  - 외기 온도          (1)
  - 랙별 CPU 부하      (n_racks)
  - 랙별 온도          (n_racks)
  - 냉각 유닛별 현재 팬 속도    (n_cooling)
  - 냉각 유닛별 현재 냉수 유량  (n_cooling)
  - 시간대 정보 (sin/cos)   (2)
  - 이전 행동 (직전 팬/냉수 설정값, [0,1])  (n_cooling*2)
  총 = 1+1+n_racks*2+n_cooling*2+2 + n_cooling*2  = 36 (기본값)

Action (연속):
  - 냉각 유닛별 팬 속도 설정값   (n_cooling) ∈ [-1, 1] → [0, 1]로 변환
  - 냉각 유닛별 냉수 유량 설정값 (n_cooling) ∈ [-1, 1] → [0, 1]로 변환
  총 = n_cooling * 2

Reward:
  에너지 절약 + 온도 유지 + 안전 제약 + 과냉각 방지 + 행동 급변 억제

  r = w_energy * r_energy
    + w_temp   * r_temp
    + w_safety * r_safety
    + r_overcool
    + w_smooth * r_smooth

  r_energy : PUE가 낮을수록(효율적) 높은 보상
             r_energy = -(PUE - 1.0)  → PUE=1이면 0, PUE=2면 -1
  r_temp   : 목표 온도(22°C)에 가까울수록 높은 보상
             r_temp = -|T_room - T_target| / T_range
  r_safety : 안전 온도 범위 이탈 시 큰 패널티
             r_safety = -10 if T > T_max else 0
  r_smooth : 직전 행동 대비 변화량(L1)에 비례한 패널티 (급변 억제)
             r_smooth = -mean(|a_t - a_{t-1}|)

에피소드: 24시간 (288 step × 5분)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from simulator import DataCenterSimulator


class DataCenterCoolingEnv(gym.Env):
    """
    데이터센터 냉각 제어 환경.
    연속 행동 공간 → SAC, TD3 등 Actor-Critic 알고리즘 적합.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_racks:    int   = 10,
        n_cooling:  int   = 3,
        t_target_c: float = 22.0,
        t_max_c:    float = 35.0,
        episode_len: int  = 288,       # 24시간 × 5분 간격
        render_mode: Optional[str] = None,
        # 보상 가중치
        w_energy:   float = 1.0,
        w_temp:     float = 2.0,
        w_safety:   float = 5.0,
        w_smooth:   float = 0.5,
    ):
        super().__init__()
        self.n_racks     = n_racks
        self.n_cooling   = n_cooling
        self.t_target    = t_target_c
        self.t_max       = t_max_c
        self.t_min       = 18.0
        self.episode_len = episode_len
        self.render_mode = render_mode
        self.w_energy    = w_energy
        self.w_temp      = w_temp
        self.w_safety    = w_safety
        self.w_smooth    = w_smooth

        self.sim = DataCenterSimulator(
            n_racks=n_racks, n_cooling=n_cooling
        )

        # ── 행동 공간: 팬 속도 + 냉수 유량 (각 [-1,1], 변환 후 [0,1])
        act_dim = n_cooling * 2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # ── 상태 공간: 기존 30차원 + 이전 행동(act_dim) = 36차원
        #    이전 행동을 관측에 넣어 정책이 직전 제어값 대비 변화를 인지하게 함.
        #    정규화 값이 ±3을 넘을 수 있어 범위를 [-inf, inf]로 둔다(클리핑 버그 제거).
        obs_dim = 1 + 1 + n_racks * 2 + n_cooling * 2 + 2 + act_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 에피소드 통계 / 이전 행동(정규화 [0,1])
        self._step        = 0
        self._ep_pue_list = []
        self._last_info   = {}
        self._prev_action_01 = np.full(act_dim, 0.5, dtype=np.float32)

    # ──────────────────────────────────────────
    # reset
    # ──────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # options={"scenario": "high"} 식으로 특정 시나리오를 강제할 수 있음.
        # 지정하지 않으면 시뮬레이터가 4종 중 무작위 선택.
        scenario = (options or {}).get("scenario")
        self.sim.reset(seed=seed, scenario=scenario)
        self._step        = 0
        self._ep_pue_list = []
        self._prev_action_01 = np.full(self.n_cooling * 2, 0.5, dtype=np.float32)

        obs  = self._get_obs()
        info = {"step": 0, "pue": 1.5, "room_temp_c": self.sim.room_temp_c,
                "scenario": self.sim.scenario}
        return obs, info

    # ──────────────────────────────────────────
    # step
    # ──────────────────────────────────────────
    def step(self, action: np.ndarray):
        # [-1,1] → [0,1] 변환
        action_01 = (np.clip(action, -1, 1) + 1.0) / 2.0

        fan_speeds   = action_01[:self.n_cooling]
        water_flows  = action_01[self.n_cooling:]

        # 행동 급변 억제 패널티: 직전 행동 대비 변화량(L1)
        r_smooth = -float(np.mean(np.abs(action_01 - self._prev_action_01)))

        # 물리 시뮬레이션 진행
        sim_info = self.sim.step_physics(fan_speeds, water_flows)
        self._step += 1
        self._ep_pue_list.append(sim_info["pue"])

        # 보상 계산 (에너지+온도+안전+과냉각 가중합 + 행동 급변 억제)
        reward = self._compute_reward(sim_info) + self.w_smooth * r_smooth

        # 다음 관측에 넣을 '이전 행동' 갱신
        self._prev_action_01 = action_01.astype(np.float32)

        # 종료 조건
        terminated = False
        truncated  = self._step >= self.episode_len

        # 온도 위험 수준이면 즉시 종료
        if sim_info["room_temp_c"] > self.t_max + 5.0:
            terminated = True
            reward    -= 20.0

        obs  = self._get_obs()
        info = {
            "step":           self._step,
            "pue":            sim_info["pue"],
            "room_temp_c":    sim_info["room_temp_c"],
            "outdoor_temp_c": sim_info["outdoor_temp_c"],
            "it_power_kw":    sim_info["it_power_kw"],
            "cooling_power_kw": sim_info["cooling_power_kw"],
            "avg_pue":        float(np.mean(self._ep_pue_list)),
            "fan_speeds":     fan_speeds.tolist(),
            "water_flows":    water_flows.tolist(),
            "scenario":       self.sim.scenario,
        }
        self._last_info = info

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────
    # 보상 함수
    # ──────────────────────────────────────────
    def _compute_reward(self, sim_info: dict) -> float:
        pue      = sim_info["pue"]
        t_room   = sim_info["room_temp_c"]
        t_range  = self.t_max - self.t_min

        # 1) 에너지 효율 보상: PUE 최소화
        #    PUE 1.0 → r=0, PUE 2.0 → r=-1.0
        r_energy = -(pue - 1.0)

        # 2) 온도 유지 보상: 목표 온도에 가까울수록
        t_err    = abs(t_room - self.t_target) / t_range
        r_temp   = -t_err

        # 3) 안전 제약: 위험 온도 진입 시 큰 패널티
        if t_room > self.t_max:
            r_safety = -10.0 * (t_room - self.t_max)
        elif t_room < self.t_min:
            r_safety = -5.0 * (self.t_min - t_room)
        else:
            r_safety = 0.0

        # 4) 과냉각 패널티 (불필요한 에너지 낭비)
        excess_cooling = max(
            0,
            sim_info["total_cooling_kw"] - sim_info["total_heat_kw"] * 1.2
        )
        r_overcool = -0.01 * excess_cooling

        reward = (
            self.w_energy * r_energy
            + self.w_temp   * r_temp
            + self.w_safety * r_safety
            + r_overcool
        )
        return float(reward)

    # ──────────────────────────────────────────
    # 관찰 벡터
    # ──────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        sim = self.sim

        # 온도 정규화 (평균 25°C, std 8°C 가정)
        t_room_norm    = (sim.room_temp_c    - 25.0) / 8.0
        t_outdoor_norm = (sim.outdoor_temp_c - 15.0) / 10.0

        # 랙 CPU 부하 [0,1] → 그대로
        cpu_loads = np.array([r.cpu_load for r in sim.racks], dtype=np.float32)

        # 랙 온도 정규화
        rack_temps = np.array(
            [(r.temp_c - 25.0) / 10.0 for r in sim.racks], dtype=np.float32
        )

        # 냉각 유닛 현재 설정값 [0,1]
        fan_speeds   = np.array([c.fan_speed          for c in sim.coolers], dtype=np.float32)
        water_flows  = np.array([c.chilled_water_flow for c in sim.coolers], dtype=np.float32)

        # 시간대 인코딩 (24시간 주기)
        hour_frac = (self._step * 5 / 60) % 24 / 24.0
        time_sin  = np.sin(2 * np.pi * hour_frac)
        time_cos  = np.cos(2 * np.pi * hour_frac)

        obs = np.concatenate([
            [t_room_norm, t_outdoor_norm],
            cpu_loads,
            rack_temps,
            fan_speeds,
            water_flows,
            [time_sin, time_cos],
            self._prev_action_01,          # 이전 행동 (act_dim) → 36차원
        ]).astype(np.float32)

        return obs

    def render(self):
        i = self._last_info
        if i:
            print(
                f"Step {i['step']:>3} | "
                f"온도: {i['room_temp_c']:>5.1f}°C | "
                f"PUE: {i['pue']:>5.3f} | "
                f"IT: {i['it_power_kw']:>5.1f}kW | "
                f"냉각: {i['cooling_power_kw']:>5.1f}kW"
            )


# ──────────────────────────────────────────────
# 환경 팩토리
# ──────────────────────────────────────────────
def make_env(**kwargs) -> DataCenterCoolingEnv:
    return DataCenterCoolingEnv(**kwargs)


# ──────────────────────────────────────────────
# 동작 확인
# ──────────────────────────────────────────────
if __name__ == "__main__":
    env = make_env(render_mode="human")
    obs, info = env.reset()

    print(f"Observation 차원: {obs.shape[0]}")
    print(f"Action 차원     : {env.action_space.shape[0]}")
    print(f"Action 범위     : [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print()

    # 랜덤 에이전트 1 에피소드
    done         = False
    total_reward = 0.0
    pue_list     = []

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        pue_list.append(info["pue"])
        done = terminated or truncated

    import numpy as np
    print(f"\n랜덤 에이전트 결과")
    print(f"  총 보상  : {total_reward:.2f}")
    print(f"  평균 PUE : {np.mean(pue_list):.3f}")
    print(f"  최소 PUE : {np.min(pue_list):.3f}")
    env.close()
