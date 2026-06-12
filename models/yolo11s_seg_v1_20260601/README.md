# YOLO11s-seg v1 (2026.06.01)

## 학습 정보
- 모델: YOLO11s-seg (사전학습 weights 시작)
- 데이터: 엘리베이터 CCTV (corner + center, 242 영상, 약 15,000장)
- 분할: train 11,978 / val 3,891 / test (소수)
- 클래스: 0=person, 1=dog
- Epoch: 100 (완주), Batch: 32, imgsz: 640
- GPU: RTX A6000 × 2 (DDP)
- 학습 시간: 약 1시간 20분

## 최종 성능 (검증셋 기준)
- Mask mAP50: ~0.92
- Mask mAP50-95: ~0.78
- Box mAP50: ~0.92
- Box mAP50-95: ~0.81

## 비고
- 검증 학습(epoch 10)에서 mAP50 0.87 확인 후 본 학습 진행
- baseline (사전학습 YOLO11n) 대비 큰 성능 향상
- mAP50 0.92 부근에서 수렴, overfitting 없음

## 검증셋 클래스별 성능 (공식 평가, val 3,891장)

| Class | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|
| all | 0.909 | 0.818 | 0.921 | 0.785 |
| person | 0.970 | 0.932 | 0.972 | 0.884 |
| dog | 0.849 | 0.705 | 0.871 | 0.687 |

## 분석
- Person: Mask mAP50-95 0.884로 매우 정밀한 segmentation
- Dog: Mask mAP50 0.871은 양호하나 mAP50-95 0.687로 정밀도 면에서 person보다 낮음
- 원인: 클래스 불균형, dog의 작은 객체 크기, 자세 다양성으로 추정
- V2 개선 방향: dog 데이터 추가 수집, copy-paste augmentation

## 추론 속도 (RTX A6000 단일 GPU, val 100장 측정)
- 평균: 49.1 ms (FPS: 20.4)
- 최소: 19.0 ms (FPS: 52.6) — 단독 GPU 환경 추정치
- 최대: 119.6 ms
- 표준편차: 27.8 ms

## 추론 속도 분석
- 평균/최대 변동이 큰 이유: 공유 GPU 서버 환경 (gpu-106 다중 사용자)
- 본인 단독 사용 가정 시 약 20ms (50 FPS) 수준 예상
- 라벨링 도구 사용성 OK (사용자 클릭 시 거의 즉시 응답)
