# Phase 3 — SAGE → residual actuator compensator

가이드(`SO-101 SAGE + GapONet + GR00T 통합 구현 가이드.md`) §10 / `CLAUDE.md` §14 Task A–C 구현.

SAGE가 수집한 sim/real 페어 CSV를 받아, GR00T가 의도한 명령(`a_desired`)을 실제 로봇이
그 의도대로 움직이도록 보정하는 residual MLP를 학습하고 TorchScript로 export한다.
배포 루프(`so101_eval.py`)와 단위·레이트·피처가 코드 레벨에서 일치하도록 설계했다.

## 산출물

| 파일 | 역할 | 가이드 |
|---|---|---|
| `tools/feature_spec.py` | 단일 진실 소스 피처 빌더. 변환기·학습기·`actuator_compensator`가 모두 import | Task A |
| `tools/sage_to_training.py` | SAGE CSV → `train.npz`/`val.npz` (단위변환·리샘플·페어생성) | Task B |
| `tools/train_compensator.py` | residual MLP 학습 + JIT export `models/compensator.pt` + per-joint RMSE 리포트 | Task C |
| `tests/make_synthetic_sage.py` | 검증용 합성 SAGE 데이터 생성기 (산출물 아님, 테스트 픽스처) | — |

## 실제 데이터로 돌리는 법 (env C)

```bash
export PYTHONPATH="$(pwd)/tools:$PYTHONPATH"

# Phase 2에서 SAGE가 output/real/so101/custom/<motion>/ 을 채운 뒤:
python tools/sage_to_training.py \
  --sage-output output \
  --motions pick_place oscillation_low_freq random_waypoints actuator_bandwidth \
  --target-rate 30 --val-frac 0.2 --out-dir outputs

python tools/train_compensator.py \
  --data-dir outputs --epochs 200 --device cpu \
  --out models/compensator.pt
```

배포(Phase 4)에서는:
`so101_eval.py --compensate learned --compensator_model models/compensator.pt`

## 핵심 설계 결정

**1. 학습-배포 피처 정합을 코드로 강제.** 가이드가 가장 강조한 게 "train/deploy 불일치 방지"
(Task A). `feature_spec.build_features` 하나만 변환기·학습기·배포(`actuator_compensator`)가
import한다. 레이아웃을 바꾸려면 이 파일 한 곳만 고치면 되고, 나머지는 자동으로 따라간다.
`train.npz`에 `feature_dim`/`use_qdot` 메타를 박아두고 학습기가 시작 시 현재 `feature_spec`과
대조해 불일치면 즉시 에러를 낸다.

**2. qdot 정책을 플래그로.** 가이드 §14 Task A가 두 옵션을 줬다 — (A) 배포에서 Feetech
`Present_Velocity` 읽어 qdot 포함(권장), (B) qdot 미사용(가장 안전, 배포 obs에 속도 채널이
없어 유한차분 노이즈를 부르는 Risk 1 회피). `feature_spec.USE_QDOT` 한 줄로 36-dim(qdot 포함,
가이드 현행 레이아웃) ↔ 30-dim(qdot-free)을 전환한다. 현재 기본은 가이드 현행 레이아웃을
유지하기 위해 `True`(36-dim). **배포 경로에서 속도를 어떻게 줄지 팀이 정한 뒤 이 플래그를
맞추면 변환기·학습기가 자동으로 동일 레이아웃을 쓴다.** qdot-free로 가기로 하면
`USE_QDOT = False` 한 줄만 바꾸고 데이터를 다시 변환하면 된다.

**3. residual 타깃 = 측정 갭.** MVP(가이드 §14 Task B): `g = real_actual - command`를 학습하고
배포에서 `u = a_desired - g_hat`로 적용. 즉 모델은 "로봇이 명령에서 얼마나 벗어나는지"를
예측하고, 그만큼을 미리 빼준다.

**4. normalization을 모델에 fold-in.** JIT export된 `compensator.pt`는 raw 피처(`.pos` 단위)를
받아 raw residual(`.pos` 단위)을 돌려준다. 정규화 통계가 scripted 모듈 buffer로 들어가 있어
배포에서 stats 파일이나 Isaac Sim 없이 `.pt` 하나만 로드하면 된다.

**5. 단위·레이트 변환을 변환기가 전담.** SAGE는 라디안 / 그리퍼 0–1 / 50Hz, 배포는
deg / 0–100 / 30Hz. 변환기가 arm rad→deg, 그리퍼 0–1→0–100, vel rad/s→deg/s,
50Hz→30Hz 리샘플(`analysis.py`의 `np.interp` 방식)을 수행한다. actuator lag는 모델링 대상
신호이므로 제거하지 않는다(가이드 §8 Risk 2).

## 검증 결과 (합성 데이터, 알려진 gap 주입)

`tests/make_synthetic_sage.py`로 static bias `[0.8,-1.2,1.5,-0.6,0.4,2.0]`와 방향 의존
backlash를 주입한 SAGE 형식 데이터를 만들어 전체 파이프라인을 돌린 결과:

| joint | before RMSE | after RMSE | 개선 |
|---|---|---|---|
| shoulder_pan | 0.801° | 0.003° | 99.6% |
| shoulder_lift | 1.205° | 0.004° | 99.7% |
| elbow_flex | 1.507° | 0.005° | 99.7% |
| wrist_flex | 0.603° | 0.004° | 99.4% |
| wrist_roll | 0.402° | 0.003° | 99.2% |
| gripper | 1.998 | 0.003 | 99.8% |

before RMSE가 주입한 gap과 정확히 일치 → 변환기의 단위 변환이 옳다.
after RMSE가 ~0 → 학습기가 gap을 복원한다.
JIT 모델을 stats/Isaac Sim 없이 로드해 raw 피처로 예측한 gap도 주입값과 일치 → 배포 정합 확인.

> 합성 데이터는 노이즈가 작고 gap이 단순해 99%가 나온 것이다. **실제 SO-101 데이터에서는
> 이만큼 안 나온다** — hobby servo의 비선형 backlash·온도 드리프트·부하 의존성이 섞이므로
> 현실적 목표는 per-joint RMSE를 의미 있게(예: 30–60%) 줄이는 것이다.

## 다음 단계 (Phase 4 — 이 산출물에 의존)

가이드 §11 hook을 `so101_eval.py`에 결선하고, `safety_filter`를 잔차 클램프 + slew +
하드 joint-limit로 보강하며, `JOINT_LIMITS`/`MAX_DELTA`를 실제 calibration JSON 값으로
교체한다. `actuator_compensator._build_features`를 `feature_spec.build_features`를 import하도록
리팩터(현재 별도 구현이면)해 정합을 유지한다.

## 주의

- 실행 전 `export PYTHONPATH="$(pwd)/tools:$PYTHONPATH"` (세 스크립트가 `feature_spec`을 import).
- 학습은 CPU로 충분(모델이 작음). GPU0은 Isaac Sim용으로 비워둔다.
- `--motions`에 direction reversal이 많은 모션(`actuator_bandwidth`, `oscillation_low_freq`)을
  꼭 포함해야 backlash가 데이터에 드러난다(가이드 §6 Phase 2).
