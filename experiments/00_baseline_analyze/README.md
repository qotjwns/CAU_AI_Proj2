# 00_baseline_analyze

MIHBench existence에서 "모든 이미지에 X가 있는가?"를 틀리는 이유를 행동 수준에서 나눠 본다.
이미지 입력 방식과 질문 방식만 바꾼 세 조건(A/B/C)을 같은 모델·같은 문항으로 비교한다.
Hidden state, probe, fine-tuning, loss 설계는 이 실험이 끝날 때까지 보류한다.

## 조건

| 조건 | 이미지 입력 | 질문 | 최종 판단 | 재는 것 |
|---|---|---|---|---|
| **A** Independent Recognition | 한 장씩 독립 입력 (문항당 3회) | `Is there a {obj} in this image?` | 세 답의 AND | 다른 이미지 영향 없는 개별 인식 |
| **B** Contextual Recognition | 3장 동시 입력 (문항당 3회) | `Is there a {obj} in the {first/second/third} image?` | 세 답의 AND | 다중 이미지 문맥이 개별 인식에 주는 영향 |
| **C** Original MIHBench | 3장 동시 입력 | MIHBench 질문 원문 + 공식 지시문 | 모델의 Yes/No | 모델 자체의 공동 판단 |

- A, B의 프롬프트 끝에는 `\nAnswer Yes or No.`가 붙는다. 정확한 문자열은 `config.yaml`에 있다.
- C는 MIHBench 공식 입력·프롬프트·채점을 바꾸지 않는다. 지시문은 `configs/prompts.yaml`에서 가져온다.
- B의 이미지 배치는 C와 같다. 두 조건의 차이는 질문뿐이다.
- 세 조건은 체크포인트, dtype, attention, 디코딩(greedy)이 모두 같다.

차이의 의미:
- **A − B**: 다중 이미지 문맥 간섭
- **B − C**: 증거 통합 실패

## 모델별 주의

- **LLaVA-OV**: 이미지를 1장만 넣으면 프로세서가 자동으로 고해상도 anyres 타일(~3,700 토큰)로 자른다. 3장을 넣는 B·C는 이미지당 기본 해상도(730 토큰)이므로, A에서도 기본 해상도로 고정해야 이미지당 토큰 수가 같아진다 (`config.yaml`의 `anyres: false`). 구현은 1장을 다중 이미지 샘플의 한 장처럼 넣는다: 픽셀은 같은 이미지 두 장짜리 샘플의 첫 장, 이미지 토큰은 730개, 모델에는 `batch_num_images=2`를 줘서 anyres 타일을 기대하지 않게 한다 (`common/models.py`의 `LlavaOVInputs`).
- **InternVL2.5-4B는 이 실험에서 뺐다.** 3장 입력이 최대 약 10k 토큰이라 RTX 3090에서 B 조건만 약 3시간이 걸릴 것으로 보여 중간에 멈췄다. 추론 코드는 `common/models.py`에 남아 있으니 `config.yaml`의 `models`에 다시 넣으면 돌릴 수 있다. 넣는다면 공식 형식이 3장의 타일을 `<image>` 하나로 이어 붙여 이미지 경계 표시가 없으므로, B의 "first/second/third image"를 모델이 구분하지 못할 수 있음을 감안해서 해석한다.

## 파일

```
gen/
├── common/                                 공통 코드: 모델 불러오기·답과 logit(models.py), 문항 로드·클래스 추출·Yes/No 판정(mihbench.py)
├── experiments/00_baseline_analyze/        git 포함
│   ├── README.md         이 문서
│   ├── config.yaml       태스크, 모델, A/B/C 프롬프트, 디코딩, 분석 설정
│   ├── infer.py          --model <m> [--cond A B C] [--task ...] [--limit N] → raw/
│   └── analyze.py        [--model ...] raw/ + GT → samples.jsonl, 결과표, 오류 교차 분석, bootstrap CI
├── results/00_baseline_analyze/            git 제외
│   └── <model>/
│       ├── raw/                    ① 추론 원본 (infer.py). 한 번 쓰면 고치지 않는다
│       │   ├── A/<task>.jsonl      모델 호출 1번 = 1줄 (문항당 3줄)
│       │   ├── B/<task>.jsonl      모델 호출 1번 = 1줄 (문항당 3줄)
│       │   ├── C/<task>.jsonl      문항당 1줄
│       │   └── run.json            조건별 실행 설정·환경 (모델 커밋, dtype, attention, 패키지 버전, 실행 시각)
│       ├── samples.jsonl           ② 문항당 1줄: A/B/C와 GT를 합친 분석 기준 파일 (analyze.py)
│       ├── table_subset.csv        ③ 결과표 1
│       ├── table_metric.csv        ③ 결과표 2
│       └── cross.json              ③ 문항별 오류 교차표, bootstrap CI
└── figures/00_baseline_analyze/<model>/    git 제외
```

- 결과·그림의 모델별 루트 경로는 `config.yaml`의 `output`에 있다. 폴더 이름은 실험 폴더 이름과 같게 둔다. 그 아래 구조(`raw/`, `samples.jsonl`, 표)는 `infer.py`와 `analyze.py`가 맞춰 쓰는 약속이라 고정한다.
- 원본은 ① raw뿐이다. ②③은 raw와 데이터(MIHBench label, COCO 어노테이션)에서 언제든 다시 만든다. 파서나 GT 기준을 바꾸면 `analyze.py`만 다시 돌린다.
- 공통 코드는 저장소 루트의 `common/`에 둔다. 이 폴더 안의 파일은 서로 import하지 않는다 (숫자로 시작하는 폴더는 import 문으로 불러올 수 없다).

### 실행 (서버)

```bash
python experiments/00_baseline_analyze/infer.py --model qwen2.5-vl-3b --cond C --limit 20   # 짧은 확인
python experiments/00_baseline_analyze/infer.py --model qwen2.5-vl-3b                       # A, B, C 전체
```

- 호출 수: A 7,200 (1장), B 7,200 (3장), C 2,400 (3장). 끊겨도 같은 명령으로 이어서 돈다.
- 분석은 Mac에서 `python experiments/00_baseline_analyze/analyze.py`로 돌린다 (모델 불필요). A·B·C가 모두 끝난 문항만 쓰므로 추론 중에도 돌릴 수 있다.
- `config.yaml`의 세 모델(Qwen 3B, Qwen 7B, LLaVA-OV) 모두 돌릴 수 있다. 새 모델은 `--cond A B C --limit 5`로 짧게 돌려 `first_token`이 Yes/No인지 먼저 본다.

### ① raw 한 줄 (모델 호출 1번)

| 필드 | 내용 |
|---|---|
| `task`, `question_id` | 문항 |
| `image_index` | A·B만: 질문한 이미지 위치 (0, 1, 2) |
| `images` | 입력한 이미지 파일. A는 1개, B·C는 3개 |
| `obj` | 질문한 클래스 |
| `prompt` | 모델에 실제로 넣은 텍스트 |
| `response` | 응답 원문 |
| `logits` | 첫 생성 위치의 `{Yes, No, yes, no}` logit |
| `first_token` | 첫 생성 토큰 |

- 호출이 끝날 때마다 한 줄씩 바로 쓴다. 다시 실행하면 이미 있는 줄은 건너뛰고 이어서 돌린다.
- 저장된 `prompt`가 지금 config로 만든 프롬프트와 다르면 멈춘다 (조건이 다른 결과가 섞이지 않게).
- GT는 넣지 않는다.

### ② samples.jsonl 한 줄 (문항 1개)

```json
{
  "sample_id": "random_0001",
  "subset": "random",
  "model": "qwen2.5-vl-3b",
  "object": "cat",
  "image_ids": ["COCO_val2014_000000000001.jpg", "COCO_val2014_000000000002.jpg", "COCO_val2014_000000000003.jpg"],
  "image_gt": [1, 1, 0],
  "joint_gt": 0,
  "label_consistent": true,
  "A": {
    "raw_responses": ["Yes", "Yes", "No"],
    "image_preds": [1, 1, 0],
    "yes_logits": [8.2, 7.1, 1.2],
    "no_logits": [1.1, 2.3, 6.5],
    "margins": [7.1, 4.8, -5.3],
    "first_token_is_answer": [true, true, true],
    "joint_pred": 0,
    "correct": true
  },
  "B": {
    "raw_responses": ["Yes", "Yes", "Yes"],
    "image_preds": [1, 1, 1],
    "yes_logits": [7.4, 6.9, 4.2],
    "no_logits": [1.2, 1.8, 3.1],
    "margins": [6.2, 5.1, 1.1],
    "first_token_is_answer": [true, true, true],
    "joint_pred": 1,
    "correct": false
  },
  "C": {
    "raw_response": "Yes",
    "yes_logit": 7.1,
    "no_logit": 3.2,
    "margin": 3.9,
    "first_token_is_answer": true,
    "joint_pred": 1,
    "correct": false
  }
}
```

| 필드 | 정의 |
|---|---|
| `sample_id` | `<subset>_<question_id 4자리>` |
| `image_ids` | MIHBench 이미지 파일 이름, 입력 순서대로 |
| `image_gt` | MIHBench label을 따른다. label이 yes면 `[1, 1, 1]`. no면 COCO val2014 어노테이션으로 물체가 있는 이미지를 1로 표시한다. COCO로는 3장 모두 있다고 나와 어느 이미지가 0인지 알 수 없으면 `null` (이미지 단위 분석에서만 제외) |
| `joint_gt` | MIHBench 문항 label (yes = 1) |
| `label_consistent` | COCO 어노테이션만으로 계산한 정답이 MIHBench label과 같은가. 기록용이며 정답에는 영향이 없다 (참고 섹션) |
| `image_preds`, C의 `joint_pred` | 응답 원문을 공식 규칙(pgtrece/DAB)으로 읽은 값: 첫 문장에 `No`, `no`, `not` 중 하나가 있으면 0, 없으면 1 |
| A·B의 `joint_pred` | `image_preds`의 AND |
| `yes_logit(s)`, `no_logit(s)` | `logsumexp(Yes, yes)`, `logsumexp(No, no)`. 지시문의 대소문자가 모델마다 달라서 합친다 |
| `margin(s)` | `yes_logit − no_logit` |
| `first_token_is_answer` | 첫 생성 토큰이 Yes/No(대소문자 무관)인가. false면 margin이 실제 답을 반영하지 않는다 |
| `correct` | `joint_pred == joint_gt` (결과표 1·2와 같은 기준) |

## 지표

정답은 MIHBench 문항 label을 쓴다. Overall은 세 태스크 2,400문항을 합친 값이다.

- Accuracy, Precision, Recall, F1: Yes를 positive로 본다.
- False-Yes Rate = FP / (FP + TN): 정답이 No인 문항에서 Yes라고 답한 비율
- False-No Rate = FN / (FN + TP): 정답이 Yes인 문항에서 No라고 답한 비율
- Yes Ratio: 전체 문항 중 Yes라고 답한 비율
- A−B, B−C의 95% CI: 같은 문항을 짝지어 재표집하는 paired bootstrap (문항 단위, 10,000회)
- 오류 교차 분석: 문항마다 A/B/C 정오를 2×2×2로 센다. A와 B는 이미지별 GT(`image_gt`)로 이미지 단위 오류도 본다.

## 결과

### 결과표 1: Accuracy

| Subset | A ↑ | B ↑ | C ↑ | A−B | B−C |
|---|---|---|---|---|---|
| Random | — | — | — | — | — |
| Popular | — | — | — | — | — |
| Adversarial | — | — | — | — | — |
| Overall | — | — | — | — | — |

### 결과표 2: Overall 지표

| Metric | A | B | C |
|---|---|---|---|
| Accuracy | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| F1 | — | — | — |
| False-Yes Rate | — | — | — |
| False-No Rate | — | — | — |
| Yes Ratio | — | — | — |

### 해석 기준

| 결과 | 가능한 원인 |
|---|---|
| A ≈ B ≫ C | 증거 통합 문제 |
| A ≫ B ≈ C | 다중 이미지 문맥 간섭 |
| A ≫ B ≫ C | 문맥 간섭과 증거 통합 문제 모두 존재 |
| A ≈ B ≈ C (낮은 성능) | 개별 이미지 인식 문제 가능성 |
| A ≈ B ≈ C (높은 성능) | 해당 과제에서 뚜렷한 병목 없음 |

질문 형식이 조건마다 다르므로 성능 차이는 우선 행동 수준의 증거로 해석한다.

## 체크리스트

- [x] MIHBench Existence 2,400문항 준비 (`data/mihbench/questions/`, 태스크당 800)
- [x] Qwen2.5-VL-3B Vanilla 모델 준비 (`models/Qwen2.5-VL-3B-Instruct/`)
- [ ] A — 독립 이미지 추론 및 AND
- [ ] B — 다중 이미지 내 개별 질문 및 AND
- [ ] C — 공식 벤치마크 평가
- [ ] 문항별 예측 및 Yes/No Logit 저장
- [ ] Random / Popular / Adversarial 성능 비교
- [ ] Accuracy / FY / FN / F1 계산
- [ ] A−B 및 B−C 성능 차이 계산
- [ ] 문항별 오류 교차 분석
- [ ] Paired Bootstrap 95% CI 계산
- [ ] 다른 Backbone에서 재현

## 참고

- **오타 문항**: 질문 원문이 `Is there an {obj} in the imange? in all 3 images?`인 문항이 86개 있다 (random 24, popular 20, adversarial 42). C는 원문을 그대로 쓰고, A·B는 여기서 뽑은 `{obj}`로 질문을 만든다.
- **MIHBench label과 COCO가 다른 문항**: 2,400문항 중 494문항(약 21%)은 MIHBench label과 COCO 어노테이션으로 계산한 정답이 다르다. 정답은 모두 MIHBench label을 따른다.
  - label이 yes인데 COCO로는 물체가 없는 이미지가 섞인 475문항: `image_gt`를 `[1, 1, 1]`로 둔다.
  - label이 no인데 COCO로는 3장 모두 물체가 있는 19문항: 어느 이미지가 0인지 알 수 없어 `image_gt`를 `null`로 두고 이미지 단위 분석에서만 뺀다. 문항 단위 지표에는 포함한다.
  - A·B는 장마다 판단하므로 이 문항들에서 실제로는 맞게 봤어도 오답으로 잡힐 수 있다. `samples.jsonl`의 `label_consistent`로 표시해 두고 해석할 때 참고한다.
- **모델별 설정** (`config.yaml`의 `models`): 이미지 전처리와 `max_new_tokens`는 모델별 MIHBench 공식 스크립트(pgtrece/DAB)를 따르고, A·B·C는 모델마다 같은 설정을 쓴다. Qwen은 672×672 리사이즈(이미지당 576 토큰), LLaVA-OV는 672×672 기본 해상도(730 토큰)다. 정밀도는 Qwen 공식(4bit + eager) 대신 bf16 + sdpa로 돌린다. 로컬에는 qwen2.5-vl-3b만 있고 나머지는 `scripts/download_models.py`로 받는다.
- **기존 C 결과**: `results/mihbench/vanilla/qwen2.5-vl-3b-official-yn/`은 DAB 공식 스크립트로 돌린 결과다. 응답 텍스트만 있고 logit이 없어서 C는 이 실험의 파이프라인으로 다시 돌린다. 기존 Accuracy(random 0.840, popular 0.826, adversarial 0.640)는 재현 확인용으로 쓴다.
