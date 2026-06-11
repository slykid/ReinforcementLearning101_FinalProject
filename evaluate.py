"""
evaluate.py
──────────────
학습된 RL 에이전트들을 규칙 기반 베이스라인과 비교 평가.

평가 지표:
  1. 평균 PUE           : 낮을수록 효율적 (1.0이 이상적)
  2. 에너지 절감률      : vs. Fixed-50% 대비 절약 비율
  3. 온도 안전성        : 안전 범위 이탈 비율
  4. 냉각 응답성        : 부하 급변 시 온도 복원 속도

RL 에이전트 (모델 파일이 있는 것만 자동 포함):
  - SAC  : models/sac_best/best_model.zip
  - TD3  : models/td3_best/best_model.zip
  - PPO  : models/ppo_best/best_model.zip

베이스라인:
  - Rule-based : 온도에 따라 단순 규칙으로 제어
  - Fixed-50%  : 팬·냉수 모두 50% 고정 (전형적 보수적 운영)
  - Fixed-80%  : 팬·냉수 모두 80% (과냉각, 에너지 낭비)

실행:
    python evaluate.py                        # 존재하는 모든 모델 자동 평가
    python evaluate.py --episodes 10          # 에피소드 수 조정
    python evaluate.py --sac-model models/sac_best/best_model   # 경로 지정
"""

import argparse, os, platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stable_baselines3 import SAC, TD3, PPO
from env import make_env

matplotlib.use("Agg")

_kor_font = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(
    platform.system(), "NanumGothic"   # Linux: apt install fonts-nanum
)
plt.rcParams.update({"font.family": _kor_font, "axes.unicode_minus": False})

MODEL_DIR  = "models"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# RL 모델 로더
# ──────────────────────────────────────────────
_MODEL_CLASSES = {"sac": SAC, "td3": TD3, "ppo": PPO}

def _load_rl_policy(model_path: str, algo: str, env):
    """
    모델 파일이 존재하면 로드해서 predict 래퍼를 반환.
    없으면 None을 반환하고 경고만 출력 (프로그램 중단 없음).
    """
    zip_path = model_path + ".zip"
    if not os.path.exists(zip_path):
        print(f"  [skip] {algo.upper()} 모델 없음: {zip_path}")
        return None
    cls = _MODEL_CLASSES[algo]
    loaded = cls.load(model_path, env=env)
    class _Policy:
        def predict(self, obs, deterministic=True):
            return loaded.predict(obs, deterministic=deterministic)
    return _Policy()


# ──────────────────────────────────────────────
# 베이스라인 정책
# ──────────────────────────────────────────────
class FixedPolicy:
    def __init__(self, level: float, n_cooling: int = 3):
        # [-1,1] 스케일로 변환해서 반환
        val = level * 2 - 1.0
        self._action = np.full(n_cooling * 2, val, dtype=np.float32)
    def predict(self, obs, deterministic=True):
        return self._action, None


class RuleBasedPolicy:
    """
    온도 기반 단순 규칙:
    - T > 28°C  : 팬·냉수 최대 (긴급 냉각)
    - T > 25°C  : 팬·냉수 70%
    - T > 22°C  : 팬·냉수 50%
    - T <= 22°C : 팬·냉수 30% (절약 모드)
    """
    def __init__(self, n_racks=10, n_cooling=3):
        self.n_racks   = n_racks
        self.n_cooling = n_cooling

    def predict(self, obs, deterministic=True):
        # obs[0]은 정규화된 실내 온도: (T - 25) / 8
        t_room = obs[0] * 8.0 + 25.0

        if t_room > 28.0:
            level = 0.95
        elif t_room > 25.0:
            level = 0.70
        elif t_room > 22.0:
            level = 0.50
        else:
            level = 0.30

        val    = level * 2 - 1.0
        action = np.full(self.n_cooling * 2, val, dtype=np.float32)
        return action, None


# ──────────────────────────────────────────────
# 단일 에피소드 실행
# ──────────────────────────────────────────────
def run_episode(policy, env, deterministic=True, seed=None) -> dict:
    obs, _ = env.reset(seed=seed)
    done   = False
    total_reward = 0.0

    temps, pues, it_powers, cool_powers = [], [], [], []
    unsafe_steps = 0

    while not done:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        temps.append(info["room_temp_c"])
        pues.append(info["pue"])
        it_powers.append(info["it_power_kw"])
        cool_powers.append(info["cooling_power_kw"])

        if info["room_temp_c"] > env.t_max:
            unsafe_steps += 1

    return {
        "total_reward":   total_reward,
        "mean_pue":       float(np.mean(pues)),
        "min_pue":        float(np.min(pues)),
        "unsafe_rate":    unsafe_steps / max(len(temps), 1),
        "mean_temp":      float(np.mean(temps)),
        "temps":          temps,
        "pues":           pues,
        "it_powers":      it_powers,
        "cool_powers":    cool_powers,
        "total_cool_kwh": float(np.sum(cool_powers) * 5 / 60),  # kWh
        "total_it_kwh":   float(np.sum(it_powers)   * 5 / 60),
    }


# ──────────────────────────────────────────────
# 다중 에피소드 통계
# ──────────────────────────────────────────────
def evaluate_policy(policy, env, n_episodes=20, label="") -> dict:
    results = [run_episode(policy, env) for _ in range(n_episodes)]
    keys    = ["total_reward", "mean_pue", "unsafe_rate",
               "mean_temp", "total_cool_kwh"]
    stats   = {
        k: {"mean": float(np.mean([r[k] for r in results])),
            "std":  float(np.std([r[k]  for r in results]))}
        for k in keys
    }
    # 대표 에피소드: 모든 정책이 동일 시나리오를 보도록 고정 시드(seed=99)로 1회 실행
    # → 시계열 비교가 같은 부하 조건에서의 apples-to-apples 비교가 됨
    stats["sample_episode"] = run_episode(policy, env, seed=99)
    stats["label"] = label
    return stats


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
def plot_results(all_stats: list, save_dir: str, n_episodes: int = 20):
    # RL 에이전트와 베이스라인을 구분하는 고정 색상표
    # SAC=파랑, TD3=주황, PPO=초록, Rule=보라, Fixed-50=회색, Fixed-80=빨강
    _COLOR_MAP = {
        "SAC":       "#4A90D9",
        "TD3":       "#F59E0B",
        "PPO":       "#10B981",
        "Rule-based":"#9B59B6",
        "Fixed-50%": "#95A5A6",
        "Fixed-80%": "#E74C3C",
    }
    _RL_LABELS = {"SAC", "TD3", "PPO"}

    def _color(label):
        for key, c in _COLOR_MAP.items():
            if key in label:
                return c
        return "#AAAAAA"

    n = len(all_stats)
    labels = [s["label"] for s in all_stats]
    colors = [_color(lbl) for lbl in labels]

    # ── 1) 지표 비교 막대 그래프
    fig, axes = plt.subplots(1, 3, figsize=(max(14, n * 2), 5))

    for ax, metric, title in [
        (axes[0], "mean_pue",       "평균 PUE (낮을수록 좋음)"),
        (axes[1], "unsafe_rate",    "온도 이탈률 (낮을수록 좋음)"),
        (axes[2], "total_cool_kwh", "냉각 전력 소비 kWh"),
    ]:
        means = [s[metric]["mean"] for s in all_stats]
        stds  = [s[metric]["std"]  for s in all_stats]
        bars  = ax.bar(labels, means, yerr=stds, color=colors,
                       alpha=0.85, capsize=5, edgecolor="white")
        # RL 에이전트는 테두리 강조
        for bar, lbl in zip(bars, labels):
            if any(rl in lbl for rl in _RL_LABELS):
                bar.set_edgecolor("#1a1a1a")
                bar.set_linewidth(1.8)
        ax.set_title(title, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    plt.suptitle(f"정책 비교 ({len(all_stats[0]['sample_episode']['temps']) // 12}시간 × {n_episodes} 에피소드)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"비교 그래프 저장: {save_dir}/comparison.png")

    # ── 2) 대표 에피소드 시계열 비교
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    ax_temp  = fig.add_subplot(gs[0, 0])
    ax_pue   = fig.add_subplot(gs[0, 1])
    ax_power = fig.add_subplot(gs[1, 0])
    ax_cool  = fig.add_subplot(gs[1, 1])

    for stat, color in zip(all_stats, colors):
        ep  = stat["sample_episode"]
        lbl = stat["label"]
        lw  = 2.2 if any(rl in lbl for rl in _RL_LABELS) else 1.2
        ls  = "-" if any(rl in lbl for rl in _RL_LABELS) else "--"

        # 조기 종료(온도 폭주)로 에피소드 길이가 정책마다 다를 수 있으므로
        # 각 정책의 실제 길이에 맞춰 x축을 구성한다.
        steps = range(1, len(ep["temps"]) + 1)

        ax_temp.plot(steps, ep["temps"],       color=color, label=lbl, linewidth=lw, linestyle=ls)
        ax_pue.plot( steps, ep["pues"],        color=color, label=lbl, linewidth=lw, linestyle=ls)
        ax_power.plot(steps, ep["it_powers"],  color=color, label=lbl, linewidth=lw, linestyle=ls)
        ax_cool.plot( steps, ep["cool_powers"],color=color, label=lbl, linewidth=lw, linestyle=ls)

    # 목표 온도선
    ax_temp.axhline(22, color="green", linestyle=":", linewidth=1.2, alpha=0.8, label="목표 22°C")
    ax_temp.axhline(35, color="red",   linestyle=":", linewidth=1.2, alpha=0.8, label="상한 35°C")

    for ax, title, ylabel in [
        (ax_temp,  "실내 온도 추이",  "온도 [°C]"),
        (ax_pue,   "PUE 추이",        "PUE"),
        (ax_power, "IT 전력 소비",    "전력 [kW]"),
        (ax_cool,  "냉각 전력 소비",  "전력 [kW]"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step (5분 간격)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("대표 에피소드 비교 (24시간)", fontsize=13, fontweight="bold")
    plt.savefig(f"{save_dir}/timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"시계열 그래프 저장: {save_dir}/timeseries.png")


# ──────────────────────────────────────────────
# 에너지 절감률 계산
# ──────────────────────────────────────────────
def print_savings(all_stats: list):
    """Fixed-50% 대비 각 RL 에이전트의 에너지 절감률을 출력."""
    baseline = next((s for s in all_stats if "Fixed-50" in s["label"]), None)
    if not baseline:
        return

    base_cool = baseline["total_cool_kwh"]["mean"]
    base_pue  = baseline["mean_pue"]["mean"]

    rl_stats = [s for s in all_stats if any(k in s["label"] for k in ("SAC", "TD3", "PPO"))]
    if not rl_stats:
        return

    print(f"\n{'='*55}")
    print(f" 에너지 절감 요약  (기준: Fixed-50%  PUE {base_pue:.3f} / {base_cool:.1f} kWh)")
    print(f"{'='*55}")
    for s in rl_stats:
        cool   = s["total_cool_kwh"]["mean"]
        pue    = s["mean_pue"]["mean"]
        saving = (base_cool - cool) / base_cool * 100
        pue_imp = (base_pue - pue) / base_pue * 100
        print(f"  {s['label']:<6} | PUE {base_pue:.3f} → {pue:.3f} ({pue_imp:+.1f}%) | "
              f"냉각전력 {base_cool:.0f} → {cool:.0f} kWh ({saving:+.1f}% 절감)")
    print(f"{'='*55}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main(
    sac_path: str = f"{MODEL_DIR}/sac_best/best_model",
    td3_path: str = f"{MODEL_DIR}/td3_best/best_model",
    ppo_path: str = f"{MODEL_DIR}/ppo_best/best_model",
    n_episodes: int = 20,
):
    env = make_env()

    # ── RL 에이전트: 모델 파일이 있는 것만 자동 포함
    rl_candidates = [
        ("SAC", sac_path, "sac"),
        ("TD3", td3_path, "td3"),
        ("PPO", ppo_path, "ppo"),
    ]
    policies = []
    for label, path, algo in rl_candidates:
        policy = _load_rl_policy(path, algo, env)
        if policy is not None:
            policies.append((policy, label))

    if not policies:
        print("[오류] 평가할 RL 모델이 하나도 없습니다.")
        print("먼저 python train.py --algo all 을 실행하세요.")
        env.close()
        return

    # ── 베이스라인 (항상 포함)
    policies += [
        (RuleBasedPolicy(), "Rule-based"),
        (FixedPolicy(0.5),  "Fixed-50%"),
        (FixedPolicy(0.8),  "Fixed-80%"),
    ]

    print(f"\n{'='*55}")
    print(f" 정책 비교 평가 ({n_episodes} 에피소드)")
    print(f" 평가 대상: {', '.join(lbl for _, lbl in policies)}")
    print(f"{'='*55}")

    all_stats = []
    for policy, label in policies:
        print(f"\n평가: {label} ...")
        stats = evaluate_policy(policy, env, n_episodes=n_episodes, label=label)
        all_stats.append(stats)
        print(f"  평균 PUE      : {stats['mean_pue']['mean']:.4f} "
              f"± {stats['mean_pue']['std']:.4f}")
        print(f"  온도 이탈률   : {stats['unsafe_rate']['mean']*100:.2f}%")
        print(f"  냉각 소비 kWh : {stats['total_cool_kwh']['mean']:.1f}")

    print_savings(all_stats)

    print("\n시각화 생성 중...")
    plot_results(all_stats, RESULT_DIR, n_episodes=n_episodes)
    print(f"\n결과 저장: {RESULT_DIR}/")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RL 에이전트 비교 평가 — 존재하는 모델을 자동으로 포함"
    )
    parser.add_argument("--sac-model", default=f"{MODEL_DIR}/sac_best/best_model",
                        help="SAC 모델 경로 (기본: models/sac_best/best_model)")
    parser.add_argument("--td3-model", default=f"{MODEL_DIR}/td3_best/best_model",
                        help="TD3 모델 경로 (기본: models/td3_best/best_model)")
    parser.add_argument("--ppo-model", default=f"{MODEL_DIR}/ppo_best/best_model",
                        help="PPO 모델 경로 (기본: models/ppo_best/best_model)")
    parser.add_argument("--episodes",  type=int, default=20,
                        help="에피소드 수 (기본: 20)")
    # 이전 --model 플래그 하위 호환 유지
    parser.add_argument("--model", default=None,
                        help="(구버전 호환) --sac-model 과 동일")
    args = parser.parse_args()

    if args.model:
        args.sac_model = args.model

    main(
        sac_path=args.sac_model,
        td3_path=args.td3_model,
        ppo_path=args.ppo_model,
        n_episodes=args.episodes,
    )
