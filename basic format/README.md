# Basic Format

이 폴더는 AI tutorial용 공통 템플릿이다.

## 목적

- 각 AI 모델과 태스크를 같은 형식으로 정리한다.
- 논문, 구조, 중간 표현, inference 흐름을 한 번에 비교한다.
- 필요할 때 이 폴더를 복사해서 모델별 폴더로 확장한다.

## 폴더 구성

- `model.ipynb`: 모델 구조와 내부 흐름을 확인하는 노트북
- `inference.ipynb`: 추론과 시각화를 확인하는 노트북

## 사용 방식

1. 이 폴더를 복사한다.
2. 폴더 이름을 모델명으로 바꾼다.
3. `model.ipynb`에서 backbone, head, attention, query, feature flow를 확인한다.
4. `inference.ipynb`에서 실제 추론과 결과 시각화를 확인한다.
5. 필요하면 `finetune.ipynb`를 추가한다.

## 공통 체크 포인트

- 입력이 어떻게 전처리되는지
- backbone이 어떤 feature map을 내는지
- 중간 feature / attention이 어떻게 변하는지
- box, score, label이 어떻게 나오는지
- 작은 물체, 겹침, 배경 혼동에서 어디가 약한지

## 다음 단계

- DETR
- Conditional DETR
- Deformable DETR
- RT-DETR
- OWLv2
- Grounding DINO
