# VILA-M3

NVIDIA의 의료용 Vision-Language Model 실습/실험 공간.

## 개요

- 논문: *VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge* (arXiv:2411.12915)
- 기반: VILA (NVIDIA VLM) + MONAI 생태계
- 핵심 아이디어: VLM이 질의에 따라 **전문가 모델(expert model)** 을 호출(tool-calling)하고,
  그 결과(세그멘테이션/분류 출력)를 다시 컨텍스트로 받아 답변을 생성
  - 예: CT → VISTA3D 세그멘테이션, CXR → TorchXRayVision 분류, MRI → BraTS 세그멘테이션
- 모델 크기: 3B / 8B / 13B / 40B

## 계획

- [ ] 사전학습 체크포인트 로드 및 단일 이미지 추론
- [ ] expert model 호출 파이프라인 구조 분석
- [ ] 자체 데이터(CT) 적용 가능성 검토

## 구조

```
vila-m3/
├── README.md     # 이 문서
└── config.yaml   # 실험 설정 (경로, 모델, 생성 파라미터)
```
