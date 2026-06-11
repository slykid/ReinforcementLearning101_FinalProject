# 데이터센터 냉각비용 최적화 (강화학습)

물리 기반 시뮬레이터 위에서 **SAC / TD3 / PPO** 세 가지 연속제어 강화학습 알고리즘으로
데이터센터 냉각 제어 정책을 학습하고, 규칙 기반 베이스라인과 비교 평가합니다.

## 1. 개요
* AI 학습용 데이터센터가 늘면서 막대한 전력이 소비되며, 그 중 약 30~40%가 IT 장비
  발열을 제어하는 냉각 시스템에서 발생합니다.
* 기존 냉각 시스템은 관리자 경험이나 단순 임계치 휴리스틱으로 운영되어, 시시각각 변하는
  IT 부하와 외부 환경에 유연하게 대응하기 어렵습니다.
* 본 프로젝트는 복잡한 열역학 환경과 비선형 전력 패턴을 학습하여, 냉각 효율을 극대화하면서도
  서버 안전 온도를 유지하는 자율 제어 에이전트를 개발합니다.

## 2. 프로젝트 목적
* **비용·에너지 절감**: IT 부하·외기 변화를 반영해 선제적으로 냉각하여 총 냉각 전력을 최소화 (PUE↓)
* **운영 안정성 확보**: 서버 과열 다운타임을 막기 위해 온도 제약을 준수하는 강건한 정책 도출

## 3. 기대효과
* 정량적: PUE 개선, 연간 냉각 전력 운영비 절감
* 정성적: 냉각 인프라 운영 자동화, 탄소 배출 감축

## 4. MDP 문제 정의

| 구성요소 | 내용 |
|---------|------|
| **State (36차원)** | 실내온도(1) + 외기온도(1) + 랙별 CPU부하(10) + 랙별 온도(10) + 냉각유닛 팬속도(3) + 냉수유량(3) + 시간 sin/cos(2) + **이전 행동(6)** |
| **Action (6차원)** | 냉각유닛별 팬 속도(3) + 냉수 유량(3), 연속 `[-1,1]` → `[0,1]` 변환 |
| **Reward** | `1.0·r_energy + 2.0·r_temp + 5.0·r_safety + r_overcool + 0.5·r_smooth` |
| **에피소드** | 24시간 = 288 step (5분 간격) |

보상 항목:
* `r_energy = -(PUE - 1.0)` — 에너지 효율(PUE 최소화)
* `r_temp = -|T_room - 22°C| / range` — 목표 온도 유지
* `r_safety` — 안전범위(18~35°C) 이탈 시 큰 패널티
* `r_overcool` — 과냉각(불필요한 냉각) 패널티
* `r_smooth = -mean(|aₜ - aₜ₋₁|)` — 행동 급변 억제

## 5. 시뮬레이터 물리 모델 (`simulator.py`)
* 서버 랙 **10개** (8~12 kW/랙), CPU 부하에 비례해 발열 (전력의 97%가 열로 변환)
* 냉각 유닛 **3개 CRAC** (최대 50 kW): 냉각 용량 = 팬 60% + 냉수 40%
* 냉각 전력 = 팬 법칙(속도³) + 칠러 압축기(냉수 유량 비례) → **PUE 1.1~1.8** 범위 형성
* 실내 온도: 열 수지 방정식 `dT = (Q_server - Q_cooling + 외기교환) · dt / (m·c)` (dt = 5분)
* 외기 온도: 일교차 ±8°C 시뮬레이션
* **부하 시나리오 4종 (에피소드별 무작위 선택)**:
  | 시나리오 | 설명 |
  |---|---|
  | `normal` | 평일 업무 패턴 — 주간 피크 |
  | `high` | 고부하 — AI 학습·배치 작업 집중 |
  | `low` | 저부하 — 야간·주말 |
  | `variable` | 불규칙 — 랜덤 스파이크 1~3회 |

## 6. 알고리즘 (`train.py`)

연속 행동 공간에 적합한 3종을 **동일 조건(총 스텝·평가 빈도·네트워크 [256,256])**으로 비교합니다.

| 파라미터 | SAC | TD3 | PPO |
|---|---|---|---|
| 학습 방식 | Off-policy | Off-policy | On-policy |
| 탐색 | 최대 엔트로피 (α 자동) | 가우시안 노이즈 (σ=0.1) | 확률적 정책 |
| `learning_rate` | 1e-4 | 1e-4 | 3e-4 |
| `buffer_size` | 300,000 | 300,000 | — |
| `learning_starts` | 5,000 | 5,000 | — |
| `batch_size` | 256 | 256 | 64 |
| `ent_coef` | auto | — | 0.0 |
| 네트워크 | [256, 256] | [256, 256] | [256, 256] |

**평가 안정화** — `EvalEnvWrapper`는 단일 고정 시드 대신 **시드 풀(base_seed~+4, pool_size=5)을 순환**시켜,
매 평가가 동일한 다양 시나리오 집합을 보도록 합니다. → 재현성 + 시나리오 다양성을 동시에 확보.

## 7. 평가 (`evaluate.py`)

학습된 RL 에이전트(SAC/TD3/PPO)를 베이스라인과 비교합니다.

* **베이스라인**: `Rule-based`(온도 임계치 규칙), `Fixed-50%`, `Fixed-80%`(고정 출력)
* **지표**: 평균 PUE, 온도 이탈률(안전범위 이탈 비율), 냉각 전력 소비(kWh), Fixed-50% 대비 절감률
* 대표 에피소드 시계열은 **모든 정책을 동일 시드(seed=99)로 실행**해 같은 부하 조건에서 비교
* 조기 종료(온도 폭주)로 에피소드 길이가 달라도 정책별 실제 길이에 맞춰 플롯
* OS별 한글 폰트 자동 설정 (macOS `AppleGothic` / Windows `Malgun Gothic` / Linux `NanumGothic`)

## 8. 실행 방법

### 8.1 환경 준비

Python 3.13 기준이며, `requirements.txt`에 학습·평가에 실제 사용한 버전이 고정되어 있습니다.

```bash
# (1) 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# (2) 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# (선택) conda/anaconda 환경을 쓰는 경우
# conda create -n dc-cooling python=3.13 -y && conda activate dc-cooling
# pip install -r requirements.txt
```

> ⚠️ **실행 환경 주의**: 모델 `.zip` 저장/로드는 numpy / stable-baselines3 버전에 민감합니다.
> **학습과 평가는 반드시 동일한 파이썬 환경**(같은 venv 또는 같은 conda)에서 실행하세요.
> 환경이 다르면 모델 로드 시 `ModuleNotFoundError: numpy._core...` 등이 발생할 수 있습니다.
> Linux에서 한글 그래프 폰트가 깨지면 `apt install fonts-nanum` 후 실행하세요.

### 8.2 동작 확인 (학습 전 빠른 점검)

```bash
python simulator.py    # 물리 시뮬레이터: 고정 제어(팬·냉수 50%) 10 step 출력
python env.py          # Gym 환경: 랜덤 에이전트 1 에피소드 → 총보상/평균PUE 출력
```

### 8.3 학습 (`train.py`)

기본 500,000 스텝이며 `--timesteps`로 조정합니다. 산출물은 `models/`, 학습 로그는 `logs/`에 저장됩니다.

```bash
python train.py                              # SAC만 학습 (기본)
python train.py --algo td3                   # TD3만
python train.py --algo ppo                   # PPO만
python train.py --algo all                   # SAC→TD3→PPO 순차 학습 + 비교 그래프 저장
python train.py --algo all --timesteps 1000000   # 1M 스텝으로 학습
python train.py --plot                       # 학습 없이 기존 로그로 비교 그래프만 재생성
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--algo` | `sac` | `sac` / `td3` / `ppo` / `all` 중 선택 |
| `--timesteps` | `500000` | 학습 총 스텝 수 |
| `--plot` | (off) | 학습 생략, `evaluations.npz` 로 비교 그래프만 생성 |

### 8.4 평가 (`evaluate.py`)

학습된 RL 모델(존재하는 것만 자동 포함)을 베이스라인(Rule-based / Fixed-50% / Fixed-80%)과 비교합니다.

```bash
python evaluate.py                           # 기본 20 에피소드, 모든 모델 자동 평가
python evaluate.py --episodes 1000           # 1,000 에피소드로 정밀 비교
python evaluate.py --sac-model models/sac_best/best_model   # 특정 모델 경로 지정
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--episodes` | `20` | 정책당 평가 에피소드 수 |
| `--sac-model` / `--td3-model` / `--ppo-model` | `models/{algo}_best/best_model` | 평가할 모델 경로 (`.zip` 생략) |

> 평가할 모델이 하나도 없으면 먼저 `python train.py --algo all` 을 실행하라는 안내 후 종료됩니다.

### 8.5 보강 실험 (`experiments.py`) — 보고서용

다중 시드 신뢰구간(A)과 PPO learning_rate 스윕(B)을 실행합니다. **중단-안전**: 각 run이 끝날 때마다
`results/experiments_runs.jsonl` 에 즉시 기록되어, 중간에 멈춰도 다시 실행하면 완료분은 건너뛰고 이어서 진행합니다.

```bash
python experiments.py --timesteps 200000 --seeds 0 1 2   # 본 실험 (이어하기 자동)
python experiments.py --smoke                            # 빠른 동작 확인 (2k 스텝, seed 0·1)
python experiments.py --summarize-only                   # 학습 없이 요약(json)만 재생성
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--timesteps` | `200000` | run당 학습 스텝 수 |
| `--seeds` | `0 1 2` | 사용할 시드 목록 |
| `--eval-episodes` | `150` | run당 평가 에피소드 수 |
| `--smoke` | (off) | 디버그용 초경량 실행 |
| `--summarize-only` | (off) | jsonl 로그만으로 요약 재생성 |

### 8.6 학습 모니터링

```bash
tensorboard --logdir logs/                   # 브라우저에서 SAC/TD3/PPO 학습 곡선 확인
```

## 9. 산출물

> **학습된 모델 다운로드 / 재사용**: 학습이 완료된 모델(SAC / TD3 / PPO)은 저장소의 `models/`
> 디렉터리에 `.zip` 형태로 포함되어 있습니다. 저장소를 clone/다운로드하면 **별도 학습 없이 바로
> 평가에 사용**할 수 있습니다.
> ```python
> from stable_baselines3 import SAC
> model = SAC.load("models/sac_best/best_model")   # 학습된 정책 로드
> ```
> 또는 `python evaluate.py` 를 실행하면 `models/` 의 모델을 자동으로 불러와 베이스라인과 비교합니다.

| 경로 | 내용 |
|---|---|
| `models/{sac,td3,ppo}_best/best_model.zip` | 평가 기준 best 모델 |
| `models/{sac,td3,ppo}_final.zip` | 최종 스텝 모델 |
| `logs/eval_*/evaluations.npz` | 학습 중 평가 보상 추이 |
| `logs/tb_*/` | TensorBoard 로그 |
| `results/algo_comparison.png` | (train.py) 알고리즘별 학습 수렴 곡선 |
| `results/comparison.png` | (evaluate.py) 정책별 PUE·온도이탈률·냉각전력 막대 비교 |
| `results/timeseries.png` | (evaluate.py) 대표 에피소드 24시간 시계열 비교 |
| `results/experiments_runs.jsonl` | (experiments.py) run별 결과 누적 로그 — 이어하기 기준 |
| `results/experiments.json` | (experiments.py) 다중시드·lr 스윕 요약 (평균±95% CI) |

## 10. 평가 지표 정의

| 지표 | 설명 |
|------|------|
| PUE | 총 전력 / IT 전력 (1.0이 이상, 목표 ≤ 1.3) |
| 에너지 절감률 | Fixed-50% 대비 냉각 전력 절약 비율 |
| 온도 이탈률 | 안전 범위(18~35°C) 이탈 step 비율 |

## 11. 주요 코드 개선 이력

**1차 (단일 패턴 환경)**
* `simulator.py`: `cooling_power_kw`에 칠러 압축기 항 추가 → PUE 변동 폭 1.04~1.06 → 1.1~1.8로 확대 (에너지 최적화 신호 실질화)
* `train.py`: 하이퍼파라미터 튜닝(lr 3e-4→1e-4, buffer 100k→300k, starts 1k→5k, eval_freq 10k→5k), `EvalEnvWrapper`로 평가 시드 고정
* `evaluate.py`: OS별 한글 폰트 자동 설정, 마이너스 기호 깨짐 수정

**2차 (현재 — 비교 환경 강화)**
* `simulator.py`: 부하 시나리오 **4종(normal/high/low/variable)** 추가, 에피소드별 무작위 선택
* `env.py`: 관측 **30→36차원**(이전 행동 추가), obs 범위 `[-3,3]→[-∞,∞]`(클리핑 버그 제거), **`r_smooth`** 행동 급변 억제 보상 추가, `info`에 `scenario` 노출
* `train.py`: `EvalEnvWrapper`를 **시드 풀 순환**(pool_size=5)으로 변경 → 평가가 여러 시나리오를 보게 됨
* `evaluate.py`: 대표 에피소드를 동일 시드로 실행(공정 비교), 조기 종료 대응(정책별 길이 플롯), 그래프 제목 동적화

## 12. 향후 과제
* EnergyPlus 등 물리 기반 시뮬레이터 연동으로 현실성 강화
* 멀티존(구역) 확장 — 구역 간 열 흐름 모델링
* 다중 시드 실험(n≥3)으로 알고리즘 순위의 통계적 검정
* XAI 적용 — 에이전트 의사결정 설명 가능성 확보 (현장 신뢰도)
