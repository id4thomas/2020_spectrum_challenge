# 2020_spectrum_challenge
Code used in 2020 Spectrum Challenge\
https://www.spectrum-challenge.kr

## Problem 3:
### Problem Description
I/Q 데이터가 주어질 때 각 타임 슬롯마다 수신되는 신호의 변조 방식 추정\
변조 방식은 (BPSK,QPSK,16QAM,8PSK)중 한가지\

### Solution
인풋을 2개의 Conv1D 레이어를 통과 시킨후 3개의 FC레이어가 있는 분류 네트워크를 통과시킨다.
학습은 CrossEntropyLoss를 이용하며 AdamW 옵티마이저로 학습된다.

### Performance
17.13/20 Point, 86% Accuracy


## Problem 4:
### Problem Description
각 타임 슬롯 마다 신호가 전송된 서브밴드, 변조 방식 판별\

* 서브밴드: 전체 주파수 대역은 9개의 서브밴드로 구성 -> 어떤 서브밴드 인지 판별
* 변조 방식: (BPSK,QPSK,16QAM,8PSK)중 한가지

### Solution
인풋을 2개의 Conv1D 레이어를 통과 시킨후 서브밴드, 변조 방식 각각 다른 아웃풋 네트워크(clf_band, clf_mod)를 통하여 예측한다.
학습은 CrossEntropyLoss를 이용하며 AdamW 옵티마이저로 학습된다.

### Performance
30/30 Point, 100% Accuracy
