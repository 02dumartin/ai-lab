# JEPA

노트북용 축소 구현이다. 공식 체크포인트를 재현하지 않고, 네 단계가 같은 예측 구조를 어떻게 바꾸는지 코드로 읽기 위한 폴더다.

## 순서

1. `jepa.ipynb` — joint-embedding predictive architecture. context encoder, EMA target encoder, predictor.
2. `ijepa.ipynb` — 이미지 패치와 multi-block mask. Assran et al., 2023.
3. `vol_jepa.ipynb` — 3D volume용 Vol-JEPA. NeuroVFM encoder pretraining.
4. `neurovfm.ipynb` — Vol-JEPA encoder를 얼리고 Classify-Then-Aggregate diagnostic head를 학습.

구현은 `modules/`에 있다. 공식 NeuroVFM은 [MLNeurosurg/neurovfm](https://github.com/MLNeurosurg/neurovfm)이고, 여기 모델은 랜덤 초기화다. 임상 판단에 쓰지 않는다.
