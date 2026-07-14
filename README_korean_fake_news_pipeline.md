# 한국어 가짜뉴스·풍자뉴스 2단계 판독 AI

이 프로젝트는 Jupyter + Python + PyTorch 환경에서 실행하는 지도학습 파이프라인입니다.

## 파일

- `korean_fake_news_train_infer.ipynb`: 학습과 실제 판독 순서를 보여 주는 실행용 노트북
- `korean_fake_news_pipeline.py`: 데이터 처리, 세 모델, 학습 루프, 임계값 튜닝, 저장·로딩 전체 구현
- `dataset_template.csv`: `|` 구분 데이터 형식 예시. 세 행뿐이므로 학습용 데이터가 아닙니다.
- `annotation_guide.md`: LLM/사람이 근거 구간과 지수를 주석할 때 사용할 규칙
- `requirements.txt`: 권장 패키지 범위

## 최종 구조

1. **Stage 1 — 진짜뉴스 대 의심뉴스**
   - `klue/roberta-small` 기반 계층형 문서 분류기
   - 출력: `P(가짜뉴스 또는 풍자뉴스)`
   - 검증 세트 temperature scaling으로 확률을 보정하고, 의심뉴스 재현율을 우선해 임계값을 고릅니다.
   - 임계값 미만은 `진짜뉴스`, 이상은 Stage 2로 보냅니다.

2. **Stage 2-A — 감정자극·과장 지수**
   - 공유 Bi-LSTM trunk와 두 token classification head
   - subword 예측을 공백 기준 어절로 다시 합친 뒤 다음처럼 계산합니다.
   - `감정자극지수 = 감정자극 어절 수 / 전체 어절 수`
   - `과장지수 = 과장 어절 수 / 전체 어절 수`

3. **Stage 2-B — 아이러니 지수**
   - 별도의 `klue/roberta-small` 기반 회귀 모델
   - 출력: 0~1

4. **가짜/풍자 최종 규칙**
   - 과장과 아이러니가 함께 높음 → 풍자뉴스
   - 아이러니가 매우 높고 다른 지수보다 두드러짐 → 풍자뉴스
   - 감정자극만 높거나 위 조건에 해당하지 않음 → 가짜뉴스
   - 모든 경계값은 validation 데이터에서 macro-F1을 기준으로 자동 탐색합니다.

## 데이터 열

필수:

- `id`
- `text`
- `label`: `진짜뉴스`, `가짜뉴스`, `풍자뉴스`

권장/학습 신호:

- `total_attention`: 0~1
- `token_attention`: Stage 1 일반 근거 구간 JSON
- `emotion_spans`: 감정자극 단어·구문 JSON
- `exaggeration_spans`: 과장 표현 단어·구문 JSON
- `irony_spans`: 아이러니 근거 구간 JSON
- `irony_score`: 0~1
- `group_id`: 같은 사건, 원문 복제, 재작성 기사 묶음

구간 주석은 다음 두 형식을 모두 지원합니다.

```json
[{"phrase": "충격적인 진실", "score": 0.95}]
```

```json
[{"start": 18, "end": 26, "score": 0.95}]
```

실제 데이터 제작에는 두 번째 문자 위치 형식을 권장합니다. 동일한 구문이 여러 번 등장할 때의 모호성을 피할 수 있습니다.

`emotion_score`와 `exaggeration_score` 열은 선택 사항입니다. 비어 있으면 각 span과 전체 어절 수로 자동 계산합니다. 반면 `irony_score`는 문맥 수준의 연속 정답이므로 가짜/풍자 학습 행에 직접 넣어야 합니다.

## 실행

노트북 첫 셀에서 의존성을 설치한 뒤 다음 순서로 실행합니다.

```python
from korean_fake_news_pipeline import Config, run_full_training

config = Config(
    data_path="news_dataset.csv",
    output_dir="artifacts/korean_fake_news_pipeline",
)
artifacts = run_full_training(config)
```

학습이 끝난 뒤 실제 기사 판독:

```python
from korean_fake_news_pipeline import FakeNewsDetector

detector = FakeNewsDetector.load("artifacts/korean_fake_news_pipeline")
result = detector.predict("판독할 뉴스 제목과 본문", include_evidence=True)
result
```

주요 출력:

- `final_label`
- `suspicious_probability`
- `emotion_index`
- `exaggeration_index`
- `irony_index`
- `decision_reason`
- 각 모델이 높게 본 `*_evidence`

## 중요한 실험 원칙

LLM이 생성한 attention 값은 모델 내부 self-attention의 정답이나 사실성의 보증이 아닙니다. 이 코드는 해당 값을 **rationale supervision**, 즉 “판단 근거로 보아야 할 문자 구간”으로 사용합니다. 일부 표본은 사람이 교차 검수하고, 라벨·근거 주석을 만든 LLM과 최종 평가용 데이터 제작 절차를 분리하는 것이 좋습니다.

같은 기사나 같은 사건을 재작성한 문장이 train/test 양쪽에 들어가면 성능이 크게 부풀려질 수 있습니다. 가능하면 `group_id`를 작성해 그룹 단위 분할을 사용하십시오.

`dataset_template.csv`는 형식 확인 전용입니다. 세 행으로는 어떤 모델도 의미 있게 학습되지 않습니다.
