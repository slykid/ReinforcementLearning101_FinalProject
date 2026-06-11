"""
experiments.py
─────────────────
보고서용 보강 실험 (중단-안전 / 이어하기 지원):
  (A) 다중 시드 신뢰구간 — SAC/TD3/PPO × 여러 시드 → 평균 ± 95% CI
  (B) 하이퍼파라미터 비교 — PPO learning_rate 스윕 × 여러 시드

각 run이 끝날 때마다 results/experiments_runs.jsonl 에 즉시 append 한다.
중간에 멈춰도 진행분은 보존되며, 다시 실행하면 완료된 run은 건너뛰고 이어서 한다.
모든 run 완료 시 results/experiments.json (요약) 를 생성.

실행:
    python experiments.py --timesteps 200000 --seeds 0 1 2     # (이어하기 자동)
    python experiments.py --smoke                              # 빠른 동작 확인
    python experiments.py --summarize-only                     # 학습 없이 요약만 재생성
"""
import argparse, json, time, os
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

from env import make_env
from evaluate import run_episode, RuleBasedPolicy

RESULT_DIR = "results"
Path(RESULT_DIR).mkdir(exist_ok=True)
RUNS_FILE = f"{RESULT_DIR}/experiments_runs.jsonl"
SUMMARY_FILE = f"{RESULT_DIR}/experiments.json"
ALGOS = {"sac": SAC, "td3": TD3, "ppo": PPO}
EVAL_BASE_SEED = 7000


def base_params(algo, seed):
    p = dict(policy_kwargs=dict(net_arch=[256, 256]), verbose=0, seed=seed)
    if algo == "sac":
        p.update(learning_rate=1e-4, buffer_size=100_000, learning_starts=5_000,
                 batch_size=256, train_freq=1, gradient_steps=1, ent_coef="auto")
    elif algo == "td3":
        p.update(learning_rate=1e-4, buffer_size=100_000, learning_starts=5_000,
                 batch_size=256, train_freq=(1, "episode"), gradient_steps=-1)
    elif algo == "ppo":
        p.update(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, ent_coef=0.0)
    return p


def make_train_env(seed):
    env = make_env(); env.reset(seed=seed)
    return Monitor(env)


def train_one(algo, seed, timesteps, hp=None):
    params = base_params(algo, seed)
    if hp:
        params.update(hp)
    venv = DummyVecEnv([lambda: make_train_env(seed)])
    if algo == "td3":
        n = venv.action_space.shape[0]
        params["action_noise"] = NormalActionNoise(np.zeros(n), 0.1 * np.ones(n))
    model = ALGOS[algo]("MlpPolicy", venv, **params)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    venv.close()
    return model


def eval_model(model, n_episodes):
    env = make_env()
    class _P:
        def predict(self, o, deterministic=True):
            return model.predict(o, deterministic=deterministic)
    pol = _P()
    pue, cool, viol = [], [], []
    for i in range(n_episodes):
        ep = run_episode(pol, env, seed=EVAL_BASE_SEED + i)
        pue.append(ep["mean_pue"]); cool.append(ep["total_cool_kwh"]); viol.append(ep["unsafe_rate"])
    env.close()
    return float(np.mean(pue)), float(np.mean(cool)), float(np.mean(viol))


def eval_rule_reference(n_episodes):
    env = make_env(); pol = RuleBasedPolicy(); cool = []
    for i in range(n_episodes):
        cool.append(run_episode(pol, env, seed=EVAL_BASE_SEED + i)["total_cool_kwh"])
    env.close()
    return float(np.mean(cool))


def ci95(vals):
    a = np.array(vals, float); n = len(a)
    m = a.mean(); sd = a.std(ddof=1) if n > 1 else 0.0
    tval = {1: 0.0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    half = tval * sd / np.sqrt(n) if n > 1 else 0.0
    return m, sd, half


# ── 중단-안전 저장 ──────────────────────────────
def load_done():
    done = {}
    if os.path.exists(RUNS_FILE):
        for line in open(RUNS_FILE):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[r["key"]] = r
    return done


def append_run(rec):
    with open(RUNS_FILE, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_jobs(seeds):
    jobs = []
    for algo in ["sac", "td3", "ppo"]:
        for s in seeds:
            jobs.append(dict(group="A", label=algo.upper(), algo=algo, seed=s, hp=None,
                             key=f"A|{algo}|{s}"))
    for lr in [1e-4, 3e-4, 1e-3]:
        for s in seeds:
            jobs.append(dict(group="B", label=f"PPO_lr{lr:g}", algo="ppo", seed=s,
                             hp={"learning_rate": lr}, key=f"B|ppo|lr{lr:g}|{s}"))
    return jobs


def summarize(seeds):
    done = load_done()
    recs = list(done.values())
    rule = next((r["rule_cool"] for r in recs if "rule_cool" in r), None)
    if rule is None:
        rule = recs[0].get("rule_cool", 1.0) if recs else 1.0

    def agg(group):
        by = {}
        for r in recs:
            if r["group"] == group:
                by.setdefault(r["label"], []).append(r)
        out = {}
        for label, rs in sorted(by.items()):
            pue = [x["pue"] for x in rs]; cool = [x["cool"] for x in rs]
            viol = [x["viol"] for x in rs]
            save = [(rule - c) / rule * 100 for c in cool]
            mp, sp, hp = ci95(pue); ms, ss, hs = ci95(save); mc, sc, hc = ci95(cool)
            out[label] = dict(n_seeds=len(rs), seeds=sorted(x["seed"] for x in rs),
                              pue_mean=mp, pue_std=sp, pue_ci95=hp,
                              cool_mean=mc, cool_std=sc, cool_ci95=hc,
                              save_mean=ms, save_std=ss, save_ci95=hs,
                              viol_mean=float(np.mean(viol)))
        return out

    summary = dict(config=dict(seeds=seeds, rule_cool_kwh=rule, n_runs=len(recs)),
                   A_multiseed=agg("A"), B_ppo_lr=agg("B"))
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-episodes", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.timesteps, args.seeds, args.eval_episodes = 2_000, [0, 1], 5

    if args.summarize_only:
        s = summarize(args.seeds)
        print(json.dumps(s["A_multiseed"], ensure_ascii=False, indent=2)); return

    jobs = build_jobs(args.seeds)
    done = load_done()
    rule = next((r["rule_cool"] for r in done.values() if "rule_cool" in r), None)
    if rule is None:
        print(f"Rule-based 기준 계산 ({args.eval_episodes} ep)...")
        rule = eval_rule_reference(args.eval_episodes)
        print(f"  Rule-based 냉각 {rule:.1f} kWh")

    todo = [j for j in jobs if j["key"] not in done]
    print(f"전체 {len(jobs)} run 중 완료 {len(done)} / 남은 {len(todo)}\n")

    for j in todo:
        t = time.time()
        model = train_one(j["algo"], j["seed"], args.timesteps, j["hp"])
        pue, cool, viol = eval_model(model, args.eval_episodes)
        rec = dict(key=j["key"], group=j["group"], label=j["label"], algo=j["algo"],
                   seed=j["seed"], hp=j["hp"], pue=pue, cool=cool, viol=viol,
                   rule_cool=rule, timesteps=args.timesteps)
        append_run(rec)
        print(f"[{j['group']}] {j['label']:>12} seed={j['seed']}: "
              f"PUE {pue:.4f} | 냉각 {cool:.1f} | 이탈 {viol*100:.2f}% | {time.time()-t:.0f}s")

    s = summarize(args.seeds)
    print(f"\n{'='*60}\n완료 — 요약 저장: {SUMMARY_FILE}\n{'='*60}")
    print("[A] 다중 시드 평균 ± 95%CI")
    for lbl, d in s["A_multiseed"].items():
        print(f"  {lbl:>4}: PUE {d['pue_mean']:.4f}±{d['pue_ci95']:.4f} | "
              f"절감 {d['save_mean']:.1f}±{d['save_ci95']:.1f}% | 이탈 {d['viol_mean']*100:.2f}%")
    print("[B] PPO learning_rate")
    for lbl, d in s["B_ppo_lr"].items():
        print(f"  {lbl:>12}: PUE {d['pue_mean']:.4f}±{d['pue_ci95']:.4f} | 절감 {d['save_mean']:.1f}±{d['save_ci95']:.1f}%")


if __name__ == "__main__":
    main()
