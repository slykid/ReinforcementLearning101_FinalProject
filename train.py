"""
train.py
───────────────
데이터센터 냉각 에이전트 학습 — SAC / TD3 / PPO 3종 비교.

[알고리즘 비교 관점]

  PPO (On-policy)
    - 에피소드 전체를 모아 한 번에 업데이트 → 샘플 효율 낮음
    - 구현 단순, 안정적이나 수렴 속도 느림
    - 비교 포인트: Off-policy 대비 같은 step에서 성능 차이

  TD3 (Off-policy, 연속 행동)
    - 고정 가우시안 노이즈(σ=0.1)로 탐색
    - 비교 포인트: 명시적 노이즈 탐색 vs SAC의 엔트로피 탐색

  SAC (Off-policy, 최대 엔트로피)
    - 보상 최대화 + 정책 엔트로피 최대화를 동시에 추구
    - 엔트로피 계수 α 자동 튜닝 → 탐색/활용 균형 자동 조절
    - 연속 냉각 제어에 가장 적합한 구조

실행:
    python train.py                   # SAC만 학습
    python train.py --algo td3        # TD3만 학습
    python train.py --algo ppo        # PPO만 학습
    python train.py --algo all        # 3개 순차 학습 후 비교 그래프 저장
    python train.py --plot            # 기존 로그로 비교 그래프만 재생성
    python train.py --timesteps 200000 --algo all  # 전체 알고리즘 20만회 학습
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback, EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

from env import make_env

matplotlib.use("Agg")

MODEL_DIR  = "models"
LOG_DIR    = "logs"
RESULT_DIR = "results"
for d in [MODEL_DIR, LOG_DIR, RESULT_DIR]:
    Path(d).mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# 커스텀 콜백: PUE 추적
# ──────────────────────────────────────────────
class PUECallback(BaseCallback):
    """학습 중 평균 PUE를 주기적으로 출력."""

    def __init__(self, log_interval=5000, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.pue_history  = []

    def _on_step(self) -> bool:
        # Monitor 래퍼가 episode_infos를 infos에 담음
        for info in self.locals.get("infos", []):
            if "episode" in info:
                # 에피소드 종료 시 평균 PUE 기록
                ep_pue = info.get("avg_pue", None)
                if ep_pue:
                    self.pue_history.append(ep_pue)

        if self.n_calls % self.log_interval == 0 and self.pue_history:
            recent_pue = np.mean(self.pue_history[-20:])
            print(
                f"  Step {self.num_timesteps:>7,} | "
                f"최근 평균 PUE: {recent_pue:.4f}"
            )
        return True


# ──────────────────────────────────────────────
# SAC 하이퍼파라미터
# ──────────────────────────────────────────────
SAC_PARAMS = {
    "learning_rate":        1e-4,       # 3e-4 → 1e-4: 정책 업데이트 보폭을 줄여 진동 억제
    "buffer_size":          300_000,    # 100k → 300k: 더 다양한 과거 경험 유지
    "learning_starts":      5_000,      # 1k → 5k: 충분한 초기 탐색 후 학습 시작
    "batch_size":           256,
    "tau":                  0.005,      # Soft target update 비율
    "gamma":                0.99,
    "train_freq":           1,          # 매 step마다 학습
    "gradient_steps":       1,
    "ent_coef":             "auto",     # 엔트로피 계수 자동 튜닝 ← SAC 핵심
    "target_entropy":       "auto",
    "use_sde":              True,       # State Dependent Exploration: 더 효율적 탐색
    "sde_sample_freq":      4,
    "policy_kwargs":        dict(net_arch=[256, 256]),
    "verbose":              0,
    "tensorboard_log":      f"{LOG_DIR}/tb_sac/",
}

# ──────────────────────────────────────────────
# TD3 하이퍼파라미터 (비교용)
# SAC와 공정한 비교를 위해 buffer/lr/starts 동일하게 맞춤
# ──────────────────────────────────────────────
TD3_PARAMS = {
    "learning_rate":     1e-4,       # SAC와 동일
    "buffer_size":       300_000,    # SAC와 동일
    "learning_starts":   5_000,      # SAC와 동일
    "batch_size":        256,
    "tau":               0.005,
    "gamma":             0.99,
    "train_freq":        (1, "episode"),
    "gradient_steps":    -1,
    "policy_kwargs":     dict(net_arch=[256, 256]),
    "verbose":           0,
    "tensorboard_log":   f"{LOG_DIR}/tb_td3/",
}

# ──────────────────────────────────────────────
# PPO 하이퍼파라미터 (비교용)
# On-policy 특성상 buffer 없이 n_steps 단위로 수집 후 업데이트
# ──────────────────────────────────────────────
PPO_PARAMS = {
    "learning_rate":  3e-4,
    "n_steps":        2048,    # 한 번의 업데이트에 사용할 스텝 수 (1 롤아웃)
    "batch_size":     64,      # 미니배치 크기
    "n_epochs":       10,      # 같은 데이터로 반복 업데이트 횟수
    "gamma":          0.99,
    "gae_lambda":     0.95,    # GAE 추정 시 분산-편향 균형 파라미터
    "clip_range":     0.2,     # 정책 변화 폭 제한 (PPO 핵심)
    "ent_coef":       0.0,     # 엔트로피 보너스 (SAC는 자동 튜닝, PPO는 고정)
    "vf_coef":        0.5,
    "max_grad_norm":  0.5,
    "policy_kwargs":  dict(net_arch=[256, 256]),
    "verbose":        0,
    "tensorboard_log": f"{LOG_DIR}/tb_ppo/",
}


# ──────────────────────────────────────────────
# 환경 빌더
# ──────────────────────────────────────────────
# 평가 시 사용할 고정 시드 풀 크기 (= EvalCallback의 n_eval_episodes)
N_EVAL_EPISODES = 5


class EvalEnvWrapper(gym.Wrapper):
    """평가 환경: 고정된 시드 풀을 reset마다 순환 적용한다.

    단일 시드를 고정하면 평가가 항상 같은 부하 시나리오 1개만 보게 되어
    분산이 0에 가깝고(강건성 측정 불가) 특정 시나리오에 과적합된 정책을
    best로 뽑을 위험이 있다. 반대로 시드를 전혀 고정하지 않으면 RNG가
    매번 달라져 같은 정책도 보상이 크게 진동한다.

    절충: base_seed~base_seed+pool_size-1 의 고정 풀을 순환시켜,
    매 평가(n_eval_episodes회)가 동일한 다양한 시나리오 집합을 보도록 한다.
    → 재현성 + 시나리오 다양성(4종 균형)을 동시에 확보."""
    def __init__(self, env: gym.Env, base_seed: int = 99, pool_size: int = N_EVAL_EPISODES):
        super().__init__(env)
        self._seed_pool = [base_seed + i for i in range(pool_size)]
        self._idx = 0

    def reset(self, **kwargs):
        kwargs["seed"] = self._seed_pool[self._idx % len(self._seed_pool)]
        self._idx += 1
        return self.env.reset(**kwargs)


def build_env(seed=42):
    env = make_env()
    env = Monitor(env)
    return env

def build_eval_env(seed=99):
    env = make_env()
    env = EvalEnvWrapper(env, base_seed=seed, pool_size=N_EVAL_EPISODES)
    env = Monitor(env)
    return env


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────
def train_sac(total_timesteps: int = 500_000) -> SAC:
    train_env = DummyVecEnv([lambda: build_env()])
    eval_env  = DummyVecEnv([lambda: build_eval_env()])

    model = SAC("MlpPolicy", train_env, **SAC_PARAMS)

    n_actions = train_env.action_space.shape[0]
    print(f"\n{'='*55}")
    print(f" SAC 학습 시작")
    print(f" State 차원 : {train_env.observation_space.shape[0]}")
    print(f" Action 차원: {n_actions} (팬×{n_actions//2} + 냉수×{n_actions//2})")
    print(f" Timesteps  : {total_timesteps:,}")
    print(f" TensorBoard: tensorboard --logdir {LOG_DIR}/tb_sac/")
    print(f"{'='*55}\n")

    callbacks = [
        PUECallback(log_interval=5_000),
        CheckpointCallback(
            save_freq=10_000,
            save_path=f"{MODEL_DIR}/sac_checkpoints/",
            name_prefix="sac_dc",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"{MODEL_DIR}/sac_best/",
            log_path=f"{LOG_DIR}/eval_sac/",
            eval_freq=5_000,        # 10k → 5k: best model 저장 기회 2배
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )
    model.save(f"{MODEL_DIR}/sac_final")
    print(f"\nSAC 학습 완료: {MODEL_DIR}/sac_final.zip")
    return model


def train_td3(total_timesteps: int = 500_000) -> TD3:
    train_env = DummyVecEnv([lambda: build_env()])
    eval_env  = DummyVecEnv([lambda: build_eval_env()])

    # TD3는 명시적 탐색 노이즈 필요 (SAC의 엔트로피 탐색과 대비되는 핵심 차이)
    n_actions    = train_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy", train_env,
        action_noise=action_noise,
        **TD3_PARAMS,
    )

    print(f"\n{'='*55}")
    print(f" TD3 학습 시작")
    print(f" Timesteps  : {total_timesteps:,}")
    print(f" TensorBoard: tensorboard --logdir {LOG_DIR}/tb_td3/")
    print(f"{'='*55}\n")

    callbacks = [
        PUECallback(log_interval=5_000),
        EvalCallback(
            eval_env,
            best_model_save_path=f"{MODEL_DIR}/td3_best/",
            log_path=f"{LOG_DIR}/eval_td3/",
            eval_freq=5_000,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
    ]
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )
    model.save(f"{MODEL_DIR}/td3_final")
    print(f"\nTD3 학습 완료: {MODEL_DIR}/td3_final.zip")
    return model


def train_ppo(total_timesteps: int = 500_000) -> PPO:
    """
    PPO(Proximal Policy Optimization) 학습.

    On-policy 특성 때문에 SAC/TD3와 근본적으로 다르게 동작:
    - 리플레이 버퍼 없음: n_steps(=2048) 스텝을 모아 업데이트 후 폐기
    - 같은 500k 스텝에서 SAC/TD3 대비 샘플 효율이 낮을 것으로 예상
    - 그러나 안정성과 구현 단순성에서 강점
    """
    train_env = DummyVecEnv([lambda: build_env()])
    eval_env  = DummyVecEnv([lambda: build_eval_env()])

    model = PPO("MlpPolicy", train_env, **PPO_PARAMS)

    print(f"\n{'='*55}")
    print(f" PPO 학습 시작")
    print(f" Timesteps  : {total_timesteps:,}")
    print(f" n_steps    : {PPO_PARAMS['n_steps']} (On-policy 롤아웃 단위)")
    print(f" TensorBoard: tensorboard --logdir {LOG_DIR}/tb_ppo/")
    print(f"{'='*55}\n")

    callbacks = [
        PUECallback(log_interval=5_000),
        EvalCallback(
            eval_env,
            best_model_save_path=f"{MODEL_DIR}/ppo_best/",
            log_path=f"{LOG_DIR}/eval_ppo/",
            eval_freq=5_000,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
    ]
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )
    model.save(f"{MODEL_DIR}/ppo_final")
    print(f"\nPPO 학습 완료: {MODEL_DIR}/ppo_final.zip")
    return model


# ──────────────────────────────────────────────
# 알고리즘 비교 시각화
# ──────────────────────────────────────────────
def plot_comparison(log_dir: str = LOG_DIR, result_dir: str = RESULT_DIR):
    """
    SAC / TD3 / PPO 의 평가 보상 곡선을 한 그래프에 겹쳐 그림.
    EvalCallback 이 저장한 evaluations.npz 파일을 읽어 사용.
    """
    import platform
    _font = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(
        platform.system(), "NanumGothic"
    )
    plt.rcParams.update({"font.family": _font, "axes.unicode_minus": False})

    configs = [
        ("SAC", f"{log_dir}/eval_sac/evaluations.npz", "#00B4D8", "-"),
        ("TD3", f"{log_dir}/eval_td3/evaluations.npz", "#F59E0B", "--"),
        ("PPO", f"{log_dir}/eval_ppo/evaluations.npz", "#10B981", "-."),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    found_any = False

    for label, path, color, ls in configs:
        if not os.path.exists(path):
            print(f"  [skip] 로그 없음: {path}")
            continue
        found_any = True
        data      = np.load(path)
        timesteps = data["timesteps"]               # (n_evals,)
        results   = data["results"]                 # (n_evals, n_episodes)
        means     = results.mean(axis=1)
        stds      = results.std(axis=1)

        ax.plot(timesteps, means, label=label, color=color,
                linewidth=2.2, linestyle=ls)
        ax.fill_between(timesteps,
                        means - stds, means + stds,
                        alpha=0.12, color=color)

    if not found_any:
        print("비교할 로그 파일이 없습니다. 먼저 학습을 실행하세요.")
        plt.close()
        return

    ax.set_xlabel("학습 스텝 수", fontsize=12)
    ax.set_ylabel("평균 에피소드 보상 (±1σ)", fontsize=12)
    ax.set_title("알고리즘 비교: 학습 수렴 곡선", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = f"{result_dir}/algo_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n비교 그래프 저장: {out}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
_TRAINERS = {
    "sac": train_sac,
    "td3": train_td3,
    "ppo": train_ppo,
}

def main(total_timesteps: int = 500_000,
         algo: str = "sac",
         plot_only: bool = False):

    if plot_only:
        print("기존 로그에서 비교 그래프 생성 중...")
        plot_comparison()
        return

    targets = list(_TRAINERS.keys()) if algo == "all" else [algo]

    for name in targets:
        print(f"\n{'━'*55}")
        print(f"  [{name.upper()}] 학습 시작  ({total_timesteps:,} steps)")
        print(f"{'━'*55}")
        _TRAINERS[name](total_timesteps)

    if len(targets) > 1:
        print("\n모든 알고리즘 학습 완료 — 비교 그래프 생성 중...")
        plot_comparison()

    print("\n학습 완료!")
    print("  단일 정책 평가: python evaluate.py")
    print("  비교 그래프   : python train.py --plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="데이터센터 냉각 강화학습 — SAC / TD3 / PPO 학습"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="학습 총 스텝 수 (기본값: 500,000)",
    )
    parser.add_argument(
        "--algo", type=str, default="sac",
        choices=["sac", "td3", "ppo", "all"],
        help="학습할 알고리즘 (all: 3개 순차 학습 후 비교 그래프 저장)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="학습 없이 기존 로그로 비교 그래프만 재생성",
    )
    # 이전 --compare 플래그 하위 호환 유지
    parser.add_argument("--compare", action="store_true",
                        help="(구버전 호환) --algo all 과 동일")
    args = parser.parse_args()

    if args.compare:
        args.algo = "all"

    main(total_timesteps=args.timesteps, algo=args.algo, plot_only=args.plot)
