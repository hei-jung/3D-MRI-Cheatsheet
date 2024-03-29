# 3D-MRI-Cheatsheet
3D MRI 영상을 다루면서 자주 쓰는 코드 모음

### 목차
1. [자주 쓰는 라이브러리](#자주-쓰는-라이브러리)
2. [이미지 전처리](#이미지-전처리)
3. [레이블 전처리](#레이블-전처리)
4. [학습 관련](#학습-관련)
5. [Pretrained Model](#Pretrained-Model)
6. [Figure and Table](#Figure-and-Table)
7. [기타 잡다한 코드](#기타-잡다한-코드)


## 자주 쓰는 라이브러리

> pydicom

DICOM 이미지를 다룰 수 있다.

```console
pip install pydicom
```

## 이미지 전처리

> 3D 이미지를 numpy 배열로 저장
```python
# source reference: https://pydicom.github.io/pydicom/dev/auto_examples/image_processing/reslice.html
df = pd.read_csv('./data.csv', index_col=0)  # index column contains folder names in my data
images = []
for fname in df.index:
    img_dir = os.listdir(fname)
    files = []
    for img in img_dir:
        files.append(dcmread(fname + '/' + img))
    
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount += 1
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    image = np.zeros(img_shape)
    
    for i, ds in enumerate(slices):
        data = ds.pixel_array
        image[:,:,i] = data
    images.append(image)
arr = np.array(images)  # to numpy array
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
img[img < _] = _
img[img > _] = _
```

> 2D 이미지 resizing
```python
from skimage.transform import resize
#...
img = resize(img, (NEW_WIDTH, NEW_HEIGHT), anti_aliasing=True)
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
ds.PixelData = resized_img.tobytes()
ds.Rows, ds.Columns = resized_img.shape ### !!!!!!!!!!
ds.save_as(FILENAME)
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

> single column for label
```python
class MyDataset(Dataset):
    
    def __init__(self, df=None, col=0):
        if df is None:
            df = load_data()
        self.X = df['image'].values
        self.y = df.iloc[:, col]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = np.array([self.y.iloc[idx]]).astype('float')
        return [image, label]
```

> multiple columns for label
```python
class MyDataset(Dataset):
    
    def __init__(self, df=None):
        if df is None:
            df = load_data()
        self.X = df['image'].values
        self.y = df.iloc[:, :5]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = np.array(self.y.iloc[idx]).astype('float')
        return [image, label]
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

### 학습

> required libraries
```python
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
```

> basic settings
```python
dataset = MyDataset()  # load dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device

# set loss functions
criterion1 = nn.L1Loss()
criterion2 = nn.MSELoss()
```

> K-fold Cross Validation (k=10)
```python
# source reference: https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
k = 10
splits = KFold(n_splits=k, shuffle=True, random_state=42)
foldperf = {}

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('Fold {}'.format(fold+1))
    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(dataset, batch_size=2, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=2, sampler=test_sampler)
    
    # model = resnet(50, in_channels=1, num_classes=5)  # model_depth = [10, 18, 34, 50, 101, 152, 200]
    # model = inception_v4(num_classes=5, in_channels=1)
    model = inception_resnet_v2(num_classes=5, in_channels=1)
    # model = densenet(121, in_channels=1, num_classes=5)  # model_depth = [121, 169, 201, 264]

    model.load_state_dict(torch.load('{}.pth'.format(type(model).__name__)))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) ##1e-3
    
    model = model.float()
    history = {'train_mae':[], 'test_mae':[], 'train_mse':[]}
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, criterion1, criterion2, optimizer, train_loader)
        test_loss = valid_epoch(model, criterion1, test_loader)
        
        train_mae = train_loss[0] / len(train_loader.sampler)
        train_mse = train_loss[1] / len(train_loader.sampler)
        test_mae = test_loss / len(test_loader.sampler)
        
        print("\nEpoch:{}/{} AVG Training MAE Loss:{:.3f} AVG Test MAE Loss:{:.3f}".format(epoch+1,
                                                                                num_epochs,
                                                                                train_mae,
                                                                                test_mae))
        print(" AVG Training MSE Loss:{:.3f}".format(train_mse))
        
        history['train_mae'].append(train_mae)
        history['test_mae'].append(test_mae)
        history['train_mse'].append(train_mse)
    print()
    foldperf['fold{}'.format(fold+1)] = history
```

> without Cross Validation
```python
# split data to train and validation sets
train, test = train_test_split(dataset, test_size=test_size)  # test_size default 0.25

train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)

losses = {'train_mae': [], 'test_mae': [], 'train_mse': []}

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.float()

for epoch in range(num_epochs):
    train_loss = train_epoch(model, criterion1, criterion2, optimizer, train_dl)
    test_loss = valid_epoch(model, criterion1, test_dl)
    
    train_mae = train_loss[0] / len(train_dl)
    train_mse = train_loss[1] / len(train_dl)
    test_mae = test_loss / len(test_dl)
    
    print("\nEpoch:{}/{} AVG Training MAE Loss:{:.3f} AVG Test MAE Loss:{:.3f}".format(epoch+1,
                                                                                num_epochs,
                                                                                train_mae,
                                                                                test_mae))
    print(" AVG Training MSE Loss:{:.3f}".format(train_mse))
        
    losses['train_mae'].append(train_mae)
    losses['test_mae'].append(test_mae)
    losses['train_mse'].append(train_mse)
```

### Re-train

1. retrain method

```python
from torch.autograd import Variable
def retrain(trainloader, model, epoch, criterion, optimizer, use_cuda=True):
    model.train()
    total = 0, 0
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        if use_cuda:
            data, target = data.float(), target.float()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))

    loss_avg = loss_sum / len(trainloader.dataset)
    print()
    print('Train Epoch: {}\tAverage Loss: {:.3f}'.format(epoch, loss_avg))
```

```python
retrain(train_loader, model, 2, criterion, optimizer)
```

2. w/o method

```python
test_model.cuda()
total_step = len(train_loader)
loss_list = []
num_epochs = 2

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, torch.float)
        
        # Run the forward pass
        outputs = test_model(inputs)
        print(outputs.shape)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("\nEpoch [{} / {}], Step [{} / {}], Loss: {:4f}\n".format(epoch+1, num_epochs, i+1, total_step, loss.item()))
```

### 데이터 시각화

> loss plot
```python
def plot_loss(train_losses, valid_losses, title='model loss'):
    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.title(title)
    plt.ylabel('loss'); plt.xlabel('epoch')
    if valid_losses==[]:  # if there are only train losses
        plt.show()
    else:
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
```

> prediction and answer plot
```python
plt.figure(figsize=(15, 15))  # configure figure size

# get prediction
y = {'label': [], 'pred':[]}
for i in range(len(test)):
    image = np.zeros((1, 1, 232, 224, 224))
    image[0, :, :, :, :] = test[i][0]
    x = torch.cuda.FloatTensor(image)
    pred = model(x)
    pred = pred.detach().cpu().numpy()
    y['label'].append(denormalize(test[i][1][0], data[data.columns[0]]))
    y['pred'].append(denormalize(pred[0][0], data[data.columns[0]]))

## for multiple labels
# for i in range(5):
#     plt.subplot(3, 2, i+1)

# linear y=x
x_new = np.linspace(0, max(y['pred']))
y_new = x_new
plt.plot(x_new, y_new, c='r')

# scatter plot predictions
plt.scatter(y['label'], y['pred'], c='b')
plt.axis('square')  # prettier
plt.ylabel('prediction'); plt.xlabel('ground truth')  # set axis name

plt.tight_layout()  # prettier
# set x, y limits if needed
plt.xlim([0,1])
plt.ylim([0,1])

plt.show()
```

## Pretrained Model

> 내 학습 모델 `.pth` 파일로 저장
```python
torch.save(model.state_dict(), '{}.pth'.format(type(model).__name__)) # save model as class name
```

> `.pth` 파일 불러오기
```python
model = MyModel(in_channels, num_classes)
model.load_state_dict(torch.load('{}.pth'.format(type(model).__name__)) # ...because I saved my model as class name
# parameters like in_channels, num_classes must match
```

> to change num of classes (ex. inception-resnet)
```python
_model = inception_resnet_v2()  # default
_model.load_state_dict(torch.load('InceptionResnetV2.pth'))
num_ftrs = _model.fc.in_features
_model.fc = nn.Linear(num_ftrs, new_num_channels)
model = _model
```

### 에러 잡기

> Error(s) in loading state_dict
>> Missing key(s) in state_dict & Unexpected key(s) in state_dict
```python
model.load_state_dict(pretrained, strict=False)  # strict: False로 해주기
```

## Figure and Table

### model summary

```python
from models.custom_net import *
import numpy as np
import os
from torchsummary import summary
model = CustomNet(classes=1)
img = np.load('img_npy/' + os.listdir('img_npy/')[0])
shape = [1]
shape += list(img.shape)
summary(model.cuda(), tuple(shape)) #####
```

### figure number

```python
i = 0
plt.figure(figsize=(16,8))
for key, value in samples.items():
    filename = img_nifti + value.index[0] + '/'
    filename = filename + os.listdir(filename)[0]
    
    img = nib.load(filename)
    img = img.get_fdata()
        
    mip = np.max(img, axis=1)
    mip = zoom(mip, (1,512/mip.shape[1]))
    mip = rotate(mip, 90)

    plt.subplot(2,4,i+1)
    plt.text(10,10, chr(ord('A')+i),
           horizontalalignment='left',
           verticalalignment='top',
           bbox=dict(facecolor='white', alpha=0.6),
           fontsize=12.5)
    plt.imshow(mip, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2,4,i+5)
    plt.text(10,10, chr(ord('A')+i),
       horizontalalignment='left',
       verticalalignment='top',
       bbox=dict(facecolor='white', alpha=0.6),
       fontsize=12.5)
    axial = img[:, :, img.shape[2]//2]
    axial = rotate(axial, 90)
    plt.imshow(axial, cmap='gray')
    plt.axis('off')
    i+=1
plt.tight_layout()
plt.show()
plt.close()
```

## 기타 잡다한 코드

### cuda 캐시 비우기
```python
# CUDA out of memory
import gc
gc.collect()
torch.cuda.empty_cache()
```

### 총 걸린 시간 계산
```python
import time
start = time.time()
##
end = time.time()
exec_time = end - start  # return seconds
```

### Numpy 배열 관련

> 배열 순서 바꾸기

ex. NHWC에서 pytorch에서 지원하는 NCHW 형태로 바꾸고 싶을 때
```python
# if x.shape is (N, H, W, C)
x = x.transpose(0,3,1,2) # 또는
x = np.transpose(x[:,:,:,:], (0,3,1,2))
```

> 파일로 저장
```python
x = np.array([[1,2], [3,4]])
np.save(FILENAME, x)
```

> 배열 파일 불러오기
```python
x = np.load('{FILENAME}.npy')
```

### Pandas Dataframe 관련

> 파일 불러오기
```python
# csv 파일의 경우
df = pd.read_csv(FILENAME)

# 불러올 때 index 지정
df = pd.read_csv(FILENAME, index_col=COLUMN_NUMBER)
```

> 행 index, 열 index 이름 보기
```python
# rows
df.index

# columns
df.columns
```

> 한 컬럼 기준 크기순 정렬
```python
df_sorted = df.sort_values(COLUMNNAME)
```

> 컬럼 선택
```python
# single column
df_col = df[COLUMNNAME]

# multiple columns
useful_columns = ['col1', 'col2', 'col3']
df_col = df[useful_columns]
```

> loc, iloc
```python
useful_columns = ['col1', 'col2', 'col3']
df_col = df.loc[:, useful_columns]

useful_rows = ['row1', 'row2', 'row3']
df_row = df.loc[useful_rows, :] # OR
df_row = df.loc[useful_rows]

df.iloc[ROW_NUM] # 행을 번호로 선택
df.iloc[:, COL_NUM] # 열을 번호로 선택

df[df['name'] == 'HJ'] # logical index
```

> 행, 열 이름 바꾸기
```python
# 행
df = df.rename(index={'old row name': 'new row name'})
# 열
df = df.rename(columns={'old column name': 'new column name'})

# 또는
df.rename(columns=COLUMNS, index=INDEX, inplace=True)
```

> 새 파일로 저장
```python
df.to_csv(FILENAME)
# index column에 이름을 주고 싶을 때
df.to_csv(FILENAME, index_label=COLUMNNAME)
# 자동 index 생성 방지
df.to_csv(FILENAME, index=False)
```

> 행 추가
```python
df = df.append(NEW_ROW, ignore_index=True) 
```

`ignore_index=True`로 하면 원래 dataframe에서 새로운 index를 할당해줌.

> 열 추가
```python
df[COLUMN_NAME] = NEW_COLUMN
```

### One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder
class_names = [[classes[0]], [classes[1]], [classes[2]]]
ohe = OneHotEncoder(sparse=False)
ohe.fit(class_names)
# check
classes_oh = ohe.transform(class_names)
print(classes_oh)
```
