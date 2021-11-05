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

> grayscale을 rgb로 변환
```python
arr = np.stack((arr,)*3, axis=-1)  # to 3 channel
```

> 이미지 자르기(가운데 기준)
```python
def crop(arr, size):
    h, w = arr.shape
    if size%2==0:
        s = size//2
        arr = arr[(h//2-s):(h//2+s), (w//2-s):(w//2+s)]
    else:
        s = (size+1)//2
        arr = arr[(h//2-s):(h//2+s-1), (w//2-s):(w//2+s-1)]
    return arr
```

> 이미지 thresholding
```python
```

> 3D 이미지 resizing
```python
def resize_data(data, new_size_x, new_size_y, new_size_z):
    initial_size_x, initial_size_y, initial_size_z = data.shape
    
    delta_x = initial_size_x / new_size_x
    delta_y = initial_size_y / new_size_y
    delta_z = initial_size_z / new_size_z
    
    new_data = np.zeros((new_size_x, new_size_y, new_size_z))
    
    for x, y, z in itertools.product(range(new_size_x), range(new_size_y), range(new_size_z)):
        new_data[x, y, z] = data[int(x*delta_x), int(y*delta_y), int(z*delta_z)]
    
    return new_data
```

> resized 이미지 새 파일로 저장
```python
```

## 레이블 전처리

> standardize(표준화)
```python
def standardize(df):  # standardization(표준화)
    new_df = df.copy()
    def _stand(x):
        mean, std = x.mean(axis=0), x.std(axis=0)
        z = (x - mean) / std
        return z
    
    for col in new_df.columns:
        new_df[col] = _stand(new_df[col])
    return new_df
```

> normalize(정규화)
```python
def normalize(df):  # 정규화
    new_df = df.copy()
    
    def _norm(x):
        z = (x - min(x)) / (max(x) - min(x))
        return z
    
    for col in df.columns:
        new_df[col] = _norm(new_df[col])
    
    return new_df
```

> denormalize
```python
def denormalize(z, x):
    x = z*(max(x) - min(x)) + min(x)
    return np.round(x, 4)
```

> destandardize + denormalize
```python
def denormalize(z, x):
    mean = x.mean(axis=0)  # x: original data
    x = z*(max(x - mean) - min(x - mean)) + min(x - mean) + mean
    return np.round(x, 4)
```

## 학습 관련

### 데이터셋 준비
```python
```

### training & validation 함수

> training
```python
def train_epoch(model, criterion, optimizer, train_loader, scheduler=None):
    # train the model
    train_loss = 0.0
    model.train()
    for batch_idx, (inputs, y) in enumerate(train_loader):
        # import data and move to gpu
        inputs, y = inputs.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
        
        # clear the gradients
        optimizer.zero_grad()
        
        # compute the model output
        y_hat = model(inputs)
        loss = criterion(y_hat, y)
        
        # credit assignment (back propagation)
        loss.backward()
        
        # update model weights
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
#             train_loss = train_loss + ((1/(batch_idx+1)) * (loss.data - train_loss))
        train_loss += loss.item() * inputs.size(0)
        print("=", end='')
    return train_loss
```

> validation
```python
def valid_epoch(model, criterion, test_loader):
    # validate the model
    valid_loss = 0.0
    model.eval()
    for batch_idx, (inputs, y) in enumerate(test_loader):
        inputs, y = inputs.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
        y_hat = model(inputs)
        loss = criterion(y_hat, y)
#             valid_loss = valid_loss + ((1/(batch_idx+1)) * (loss.data - valid_loss))
        valid_loss += loss.item() * inputs.size(0)
        print("=", end='')
    return valid_loss
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

### 에러 잡기

> Error(s) in loading state_dict
>> Missing key(s) in state_dict & Unexpected key(s) in state_dict
```python
model.load_state_dict(pretrained, strict=False)  # strict: False로 해주기
```

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
