# 3D-MRI-Cheatsheet
3D MRI 영상을 다루면서 자주 쓰는 코드 모음

### 목차
1. [자주 쓰는 라이브러리](#자주-쓰는-라이브러리)
2. [이미지 전처리](#이미지-전처리)
3. [레이블 전처리](#레이블-전처리)
4. [학습 관련](#학습-관련)
5. [Pretrained Model](#Pretrained-Model)
6. [기타 잡다한 코드](#기타-잡다한-코드)


## 자주 쓰는 라이브러리

> pydicom

DICOM 이미지를 다룰 수 있다.

```console
pip install pydicom
```

## 이미지 전처리

> 3D 이미지를 numpy 배열로 저장
```python
```

> 이미지 자르기
```python
```

> 이미지 thresholding
```python
```

> 이미지 resizing
```python
```

> resized 이미지 새 파일로 저장
```python
```

## 레이블 전처리

> standardize(표준화)
```python
```

> normalize(정규화)
```python
```

> denormalize
```python
```

> destandardize + denormalize
```python
```

## 학습 관련

### 데이터셋 준비
```python
```

### training & validation 함수

> training
```python
```

> validation
```python
```

> 학습
```python
```

> K-fold Cross Validation
```python
```

### 데이터 시각화

> loss plot
```python
```

> prediction and answer plot
```python
```

## Pretrained Model

> 내 학습 모델 `.pth` 파일로 저장
```python
```

> `.pth` 파일 불러오기
```python
````

## 기타 잡다한 코드

### cuda 캐시 비우기
```python
```

### 총 걸린 시간 계산
```python
```

### Numpy 배열 관련

> 배열 순서 바꾸기

ex. NHWC에서 pytorch에서 지원하는 NCHW 형태로 바꾸고 싶을 때
```python
```

> 파일로 저장
```python
```

> 배열 파일 불러오기
```python
```

### Pandas Dataframe 관련

> 파일 불러오기
```python
```

> 한 컬럼 기준 크기순 정렬
```python
```

> 인덱스 제거
```python
```

> 필요한 컬럼만 추리기
```python
```

> 새 파일로 저장
```python
```
