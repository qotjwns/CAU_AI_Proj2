# gen

## 프로젝트 구조

```
gen/
├── requirements.txt              패키지 버전 고정 (Python 3.11)
├── configs/
│   └── prompts.yaml              모델 계열별 텍스트 프롬프트 ({question} 자리에 질문). 추론과 학습이 같이 읽음
├── scripts/                      다운로드 스크립트 (데이터·모델 준비)
│   ├── download_coco_train2014.py
│   ├── download_mihbench.py
│   └── download_models.py
├── src/                          실험 코드
│   ├── mihbench.py               MIHBench 문항 로드, Yes/No 파싱, 공식 지표 계산 (모델과 무관)
│   ├── infer_mihbench.py         VLM으로 MIHBench 추론 후 채점
│   ├── eval_mihbench.py          저장된 응답을 다시 채점해 모델 비교 표(summary.csv)와 그래프 생성
│   ├── coco_pools.py             COCO 학습 데이터 1단계: 클래스별 P/N/어려운 N 가용 이미지 집계
│   └── prompts.py                configs/prompts.yaml 읽기·검사, 템플릿에 질문 채우기
├── data/                         데이터셋
│   ├── coco/
│   │   ├── annotations/
│   │   │   ├── instances_train2014.json
│   │   │   └── instances_val2014.json   MIHBench 이미지(val2014)의 물체 크기 확인용
│   │   └── train2014/            COCO_train2014_*.jpg 82,783장
│   ├── coco_existence/           COCO 기반 학습 데이터 (src/coco_pools.py 등이 생성)
│   │   └── pool_stats.json       클래스별 가용 이미지 수, 어려운 N 동반 클래스, 누수·물체 크기 점검
│   └── mihbench/
│       ├── questions/            태스크별 <task>.jsonl, 각 800문항
│       └── images/
│           ├── coco_val2014/     Existence, Count 이미지 1,908장
│           └── co3d/             ID Consistency 이미지 1,619장 (<카테고리>/<시퀀스>/frame*.jpg)
├── models/                       모델 체크포인트, 폴더명은 HF 저장소 이름 (예: Qwen2.5-VL-3B-Instruct/)
├── results/                      평가 결과
│   └── mihbench/
│       └── vanilla/              학습하지 않은 원본 모델 (학습 모델은 <방법>/<model>/로 추가)
│           ├── <model>/          예: qwen2.5-vl-3b-4bit/ (프롬프트는 configs/prompts.yaml, 기본은 질문 + 공식 yes/no 지시문. -4bit: 공식 Qwen 설정(4bit + eager), -sdpa: InternVL을 sdpa로 실행, -noyn: 지시문 없이 돌린 이전 결과, -official[-yn]: DAB 공식 스크립트로 돌린 Qwen 결과)
│           │   ├── <task>.jsonl  문항별 모델 응답
│           │   └── metrics.json  태스크별 accuracy, precision, recall, F1, yes-ratio + 실행 환경
│           └── summary.csv       모델 × 태스크 × subset(all, no_typo) 지표 (src/eval_mihbench.py)
└── figures/                      그래프 (results/와 같은 구조)
    └── mihbench/
        └── vanilla/              mihbench.png, mihbench_no_typo.png (src/eval_mihbench.py)
```

### 폴더별 역할

| 폴더 | 내용 | 만드는 곳 | git |
|---|---|---|---|
| `scripts/` | 데이터와 모델을 받는 스크립트. 다시 실행하면 끝난 파일은 건너뜀 | - | 포함 |
| `src/` | 추론·채점 등 실험 코드 | - | 포함 |
| `data/coco/` | COCO train2014 이미지와 instance 어노테이션 | `scripts/download_coco_train2014.py` | 제외 |
| `data/mihbench/` | MIHBench 5개 태스크 문항과 이미지 | `scripts/download_mihbench.py` | 제외 |
| `models/` | VLM 체크포인트 (커밋 고정) | `scripts/download_models.py` | 제외 |
| `results/` | 방법·모델별 응답과 지표, 비교 표 | `src/infer_mihbench.py`, `src/eval_mihbench.py` | 제외 |
| `figures/` | 방법별 그래프 | `src/eval_mihbench.py` | 제외 |

### MIHBench 태스크

| `<task>` | 이미지 폴더 |
|---|---|
| `existence_random`, `existence_popular`, `existence_adversarial` | `coco_val2014/` |
| `count` | `coco_val2014/` |
| `id_consistency` | `co3d/` |
