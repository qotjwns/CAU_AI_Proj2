# mivlm: 다중 이미지 VLM의 False-Yes 분석

Qwen2.5-VL에 이미지 3장을 주고 "이 객체가 3장 모두에 있는가?"를 물었을 때, 한 장에라도 객체가 없는데 Yes라고 답하는 오류(False-Yes)를 분석하는 연구 코드입니다. 이전 연구(STEP 1~)를 처음부터 다시 구현하고 있습니다.

지금 구현된 범위는 **데이터 준비 → 케이스 생성 → baseline 추론(STEP 1)**입니다. 지표 계산과 이후 분석(hidden state, probe, attention, intervention)은 아직 없습니다.

개발 규칙은 [CLAUDE.md](CLAUDE.md)에 있습니다.

## 실험 설계

| 항목 | 내용 |
|---|---|
| 데이터 | COCO val2017 이미지 5,000장 + COCO-ReM 주석(`instances_valrem.json`) |
| 클래스 | P 후보가 많은 순으로 20개 (person … horse) |
| P 이미지 | crowd가 아닌 해당 클래스 bbox 중 하나 이상이 이미지 면적의 5% 이상 |
| N 이미지 | 해당 클래스 주석이 크기와 상관없이 하나도 없음 |
| pool | 클래스마다 P 8장, N 8장 (seed 42) |
| pair | 클래스마다 20개(group 5개 × pair 4개), 총 400개. pair 하나는 P 3장 + N 3장 |
| 케이스 | pair마다 변형 6개, 총 2,400개 |

질문은 `Is the object category '{object}' present in all 3 images? Please answer with Yes or No.`이고, 이미지 3장 뒤에 붙습니다. 3장 모두에 있어야 Yes이므로 PPP만 정답이 Yes입니다.

| 변형 | 자리1 | 자리2 | 자리3 | 정답 | 모드 |
|---|---|---|---|---|---|
| PPP | p1 | p2 | p3 | Yes | A |
| PPN | p1 | p2 | n1 | No | A, B |
| PNN | p1 | n2 | n1 | No | A |
| NNN | n3 | n2 | n1 | No | A |
| NPP | n1 | p2 | p3 | No | B |
| PNP | p1 | n1 | p3 | No | B |

- **Mode A**는 N의 개수(0 → 3)에 따른 변화를 봅니다.
- **Mode B**는 N 1장(같은 n1)의 위치에 따른 변화를 봅니다.

## 환경 설정

Python 3.11을 씁니다. 스크립트와 실험 코드가 `mivlm` 패키지를 import하므로 `pip install -e .`까지 해야 실행됩니다.

### Mac (개발, 파일럿)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Vast.ai (GPU 서버, RTX 4090)

**인스턴스를 만들 때**

- 디스크는 40GB 이상을 권장합니다. COCO 약 2GB, Qwen2.5-VL-3B 약 7.5GB, CUDA용 torch를 포함한 가상환경이 수 GB를 차지합니다. 7B까지 받으려면 약 17GB가 더 필요합니다.
- 인스턴스를 삭제하면 그 안의 데이터도 사라집니다. 결과를 내려받기 전에는 삭제하지 마세요.

**설치**

```bash
nvidia-smi                        # 드라이버가 CUDA 13.2 이상을 지원하는지 확인
python3.11 --version              # 없으면 Python 3.11이 들어 있는 이미지를 쓰거나 따로 설치

git clone <저장소 주소> && cd CAU_AI_Proj2
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# torch와 torchvision은 CUDA 빌드를 먼저 설치한다. 그냥 requirements.txt만 설치하면 다른 빌드가 깔릴 수 있다
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
pip install -e .

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
# 기대 출력: 2.14.1+cu132 True True
```

## 데이터와 모델 준비

아래 순서대로 실행합니다. 모든 단계는 seed로 결정되므로 어느 환경에서 만들어도 같은 파일이 나와야 합니다.

```bash
python scripts/download_coco.py                   # data/coco/  이미지 5,000장 + COCO-ReM 주석
python scripts/build_pools.py                     # data/cases/pools.json   클래스 20개 x (P 8, N 8)
python scripts/build_pairs.py                     # data/cases/pairs.json   pair 400개
python scripts/build_cases.py                     # data/cases/cases.jsonl  케이스 2,400개
python scripts/download_model.py qwen2.5-vl-3b    # Hugging Face 캐시, 약 7.5GB
```

**케이스 파일 확인.** 아래 해시가 같으면 다른 환경과 같은 케이스로 실험하는 것입니다.

```bash
sha256sum data/cases/cases.jsonl      # macOS는 shasum -a 256
# 2c21311b716413ad8eaba6d361e086ea1c52864b95d4b084dca0ceb3de5a0af4
pytest tests/integration/test_cases_data.py
```

**모델 버전.** `download_model.py`는 받은 버전의 commit hash를 출력합니다. 지금까지 검증에 쓴 Qwen2.5-VL-3B의 commit은 `66285546d2b821cf421d4f5eb2576359d3770cd3`입니다. 추론 코드는 캐시에 받아 둔 버전만 쓰고, 실행마다 `meta.json`에 commit을 기록합니다.

`python scripts/view_pools.py`를 실행하면 클래스별 P/N 이미지를 `outputs/figures/pools/`에 그림으로 저장합니다(선택).

## 테스트

필요한 데이터나 모델이 없으면 해당 테스트는 건너뜁니다.

| 명령 | 개수 | 필요한 것 |
|---|---|---|
| `pytest tests/unit` | 51 | 없음. 몇 초 |
| `pytest tests/integration/test_cases_data.py` | 8 | 생성된 케이스 데이터 |
| `pytest tests/integration/test_loading_dtype.py` | 1 | InternVL3.5-1B (`download_model.py internvl3.5-1b`) |
| `pytest tests/integration/test_qwen_load.py tests/integration/test_qwen_inference.py -s` | 10 | Qwen2.5-VL-3B, 메모리 약 7.5GB |

인자 없이 `pytest`만 실행하면 모델을 로딩하는 테스트까지 모두 돕니다.

## 실행: STEP 1 baseline

```bash
python experiments/01_baseline/run.py --pilot              # 클래스별 1개, 20케이스
python experiments/01_baseline/run.py                      # 전체 2,400케이스
python experiments/01_baseline/run.py --dtype float32      # float32로 실행 (bf16 결과와 비교용)
```

| 옵션 | 내용 |
|---|---|
| `--pilot` | 클래스마다 첫 pair에서 케이스 하나씩, 변형을 돌아가며 고른 20개만 실행 |
| `--dtype float32` | 모델 전체를 float32로 로딩. 지정하지 않으면 device 기본값 |
| `--model` | `configs/models.yaml`에 등록된 이름. 기본값 `qwen2.5-vl-3b` |

- **device와 dtype:** CUDA(bf16) → MPS(bf16) → CPU(float32) 순서로 자동 선택합니다.
- **이어 돌리기:** 중간에 끊기면 같은 명령으로 다시 실행합니다. 끝난 케이스는 건너뜁니다. 기존 실행과 조건(모델 버전, dtype, 케이스 파일 등)이 다르면 멈추므로, 그때는 결과 폴더를 지우고 다시 실행합니다.
- **Vast에서:** 전체 실행은 `tmux` 안에서 돌리면 SSH 연결이 끊겨도 계속됩니다.

### 결과

`outputs/predictions/01_baseline/<모델>[_float32][_pilot]/`에 두 파일이 생깁니다.

- `meta.json`: 실행 조건. 모델 commit, device, dtype, 생성 설정, 질문 문구, 케이스 파일 해시, 라이브러리 버전, git commit
- `results.jsonl`: 케이스마다 한 줄

| 필드 | 내용 |
|---|---|
| `case_id` | `cases.jsonl`의 케이스 id. 정답과 변형은 이 id로 케이스 파일에서 찾음 |
| `raw_response` | 모델이 생성한 텍스트 (greedy, 최대 8토큰) |
| `prediction` | `raw_response`를 판정한 값. `"Yes"`, `"No"`, 둘 다 아니면 `null` |
| `yes_logit`, `no_logit` | 첫 생성 위치에서 `"Yes"`, `"No"` 토큰의 logit |
| `logit_margin` | `yes_logit - no_logit` |
| `input_token_count`, `image_token_count` | 전체 입력 토큰 수와 그중 이미지 토큰 수 |

**결과 내려받기 (Vast → Mac).** 접속 주소와 포트는 Vast 인스턴스 화면의 SSH 명령에서 확인합니다.

```bash
scp -P <포트> -r root@<주소>:<프로젝트 경로>/outputs/predictions/01_baseline ./outputs/predictions/
```

### 입력 확인

케이스 하나를 실제로 추론하고, 모델에 들어간 이미지 3장과 토큰 위치, 모델의 답과 logit을 그림 한 장으로 저장합니다.

```bash
python scripts/view_case_input.py group_01_person_1_PPN      # outputs/figures/case_inputs/
```

## 폴더 구조

```
configs/models.yaml         모델 이름 → Hugging Face repo id
scripts/                    데이터·모델 준비(download_, build_), 점검용 그림(view_)
src/mivlm/                  재사용 코드
  config.py                   프로젝트 경로, 모델 목록 읽기
  data/                       이미지 경로, 케이스 읽기, 파일럿 케이스 선택
  models/                     모델 로딩, 추론
experiments/01_baseline/    STEP 1 추론 실행
tests/unit/                 모델 없이 도는 테스트
tests/integration/          실제 데이터나 모델이 필요한 테스트
data/                       COCO와 케이스 파일 (git 제외)
outputs/                    추론 결과와 그림 (git 제외)
```

## 이전 연구와의 관계

### 맞춘 것

- **모델과 라이브러리:** Qwen/Qwen2.5-VL-3B-Instruct, torch 2.14.1, torchvision 0.29.1, transformers 5.22.2
- **입력:** user 턴 하나에 이미지 3장을 먼저, 질문을 마지막에 넣습니다. system 메시지는 넣지 않아서 chat template의 기본 문구가 들어갑니다. 이미지는 경로 문자열로 넘깁니다.
- **생성:** greedy, 최대 8토큰. `repetition_penalty`는 넘기지 않아 모델 기본값 1.05가 적용됩니다.
- **판정:** 텍스트는 앞뒤 공백과 끝의 `.!?`를 지우고 대소문자를 무시해 yes/no와 비교합니다. logit은 첫 생성 위치의 가공 전 값(`output_logits`)에서 읽습니다.
- **데이터 선정 기준:** 클래스 20개와 그 순서, 클래스별 P 후보 수(person 1,847 … horse 97)가 이전 연구와 일치합니다.

### 다른 것, 아직 확인하지 못한 것

- **뽑힌 이미지가 다릅니다.** 선정 절차와 seed는 같지만 추출 방식의 세부가 달라서 클래스별 P/N 8장이 이전 연구와 다릅니다. 결과는 "같은 절차, 다른 표본"으로 비교해야 합니다.
- **이미지 디코더.** transformers는 torchcodec 0.16 이상이 설치돼 있으면 torchcodec으로, 없으면 torchvision으로 이미지를 엽니다. 이전 연구 환경에는 torchcodec 0.16.0이 있었지만 STEP 1 당시에도 쓰였는지는 확인하지 못했습니다. 실행마다 `meta.json`의 `preprocessing.torchcodec`에 기록됩니다(`null`이면 torchvision).
- **device.** Mac(MPS)은 개발과 파일럿용입니다. 이전 연구와 수치를 비교하는 실행은 CUDA bf16에서 합니다.
- **bf16 정밀도.** logit이 25 근처일 때 bf16은 0.125 간격으로만 값을 표현합니다. margin이 0 근처인 케이스는 판정이 이 간격에 민감하므로, 같은 케이스를 `--dtype float32`로도 돌려 비교합니다.
