## MONAI 사용법

참고링크. [Load medical images](https://github.com/Project-MONAI/tutorials/blob/master/modules/load_medical_images.ipynb),
[3D image transforms](https://github.com/Project-MONAI/tutorials/blob/master/modules/3d_image_transforms.ipynb)

### LoadImage

이미지 불러오는 용도(NIfTI, DICOM, PNG format). 나의 경우 DICOM 이미지를 사용하므로 DICOM 예시를 가져와봄.

```python
import os
import shutil
import numpy as np
import itk
from PIL import Image
import tempfile
from monai.data import ITKReader, PILReader
from monai.transforms import (
    LoadImage, LoadImaged, EnsureChannelFirstd,
    Resized, EnsureTyped, Compose
)
from monai.config import print_config
```

```python
tempdir = tempfile.mkdtemp()
filename = os.path.join(tempdir, "test_image.dcm")
dcm_image = np.random.randint(256, size=(64, 128, 96)).astype(np.uint8)
itk_np_view = itk.image_view_from_array(dcm_image)
itk.imwrite(itk_np_view, filename)
```

```python
data, meta = LoadImage()(filename)

print(f"image data shape:{data.shape}")
print(f"meta data:{meta}")
```

근데 처음에 계속 `RuntimeError: can not find suitable reader for this file:` 이 에러가 났다.
tutorial에서 복붙해왔으니까 에러가 날 이유가 없는데도, 에러가 났다... 혹시 싶어서 colab으로 돌려봤는데 colab에선 에러가 안 난다.

알고 보니 requirements 일부가 설치가 안 되어있던 모양이다.

```console
$ git clone https://github.com/Project-MONAI/MONAI.git
$ cd MONAI/
$ pip install -e '.[all]'
```
출처: https://docs.monai.io/en/latest/installation.html#table-of-contents [Installation Guide]


### Load부터 Transformation까지

> Required Libraries

```python
from monai.transforms import (
    AddChanneld,
    LoadImage,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
)
import numpy as np
import matplotlib.pyplot as plt
import itk
```

> Make test image (이건 그냥 참고용)

```python
# images is list where images[0].shape = (1, 232, 224, 224)
itk_np_view = itk.image_view_from_array(images[0].astype(np.uint8))
itk.imwrite(itk_np_view, "test_image.dcm")
```

> Load Image

```python
data, meta = LoadImage()("test_image.dcm")
```

Check data:

```python
print(f"image data shape:{data.shape}")
print(f"meta data:{meta}")
```

```
image data shape:(224, 224, 232)
meta data:{'0008|0016': '1.2.840.10008.5.1.4.1.1.7.2', '0008|0018': '1.2.826.0.1.3680043.2.1125.1.21523321020922955153453371700322151', '0008|0020': '20211202', '0008|0030': '161839.797146 ', '0008|0050': '', '0008|0060': 'OT', '0008|0090': '', '0010|0010': '', '0010|0020': '', '0010|0030': '', '0010|0040': '', '0020|000d': '1.2.826.0.1.3680043.2.1125.1.47022884073917657426301854936606650', '0020|000e': '1.2.826.0.1.3680043.2.1125.1.33081996829990243804814336994781819', '0020|0010': '', '0020|0011': '', '0020|0013': '', '0020|0052': '1.2.826.0.1.3680043.2.1125.1.21082565907138371517547930251623493', '0028|0002': '1', '0028|0004': 'MONOCHROME2 ', '0028|0008': '232 ', '0028|0009': '(5200,9230)', '0028|0010': '224', '0028|0011': '224', '0028|0100': '8', '0028|0101': '8', '0028|0102': '7', '0028|0103': '0', '0028|1052': '0 ', '0028|1053': '1 ', '0028|1054': 'US', 'spacing': array([1., 1., 1.]), 'original_affine': array([[-1.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]]), 'affine': array([[-1.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]]), 'spatial_shape': array([224, 224, 232]), 'original_channel_dim': 'no_channel', 'filename_or_obj': 'test_image.dcm'}
```

```python
plt.figure("visualize", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(data[:, :, 0], cmap="gray")
plt.show()
```

> Add Channel

처음에 LoadImage 통해서 이미지 불러오면 greyscale 이미지의 경우 channel dimension이 없다.
그런데 MONAI의 transformation method들을 사용하려면 channel dimension까지 있어야 하므로, 꼭 이 과정을 거쳐야 한다. (안 그럼 에러 남)

:bangbang: | 참고로 여기서부턴 단순 numpy 배열 형태가 아니라, dictionary 형태만 인수(argument)로 넣을 수 있다.
:---: | :---

그래서 나의 경우, `data_dict`라는 dictionary 변수를 다음과 같이 만들었다.

```python
data_dict = {"image": data}
```

```python
add_channel = AddChanneld(keys=["image"])
datac_dict = add_channel(data_dict)
```

또는

```python
data_dict = AddChanneld(keys=["image"])(data_dict)
```

Check data:

```python
print(f"image shape:{data_dict['image'].shape}")  # image shape:(1, 224, 224, 232)
```

> Reorientation

axis labels: Left (L), Right (R), Posterior (P), Anterior (A), Inferior (I), Superior (S)

```python
orientation = Orientationd(keys=["image", "label"], axcodes="PLI")
data_dict = orientation(data_dict)
```

또는

```python
data_dict = Orientationd(keys=["image"], axcodes="PLI")(data_dict)
```

Check data:

```python
print(f"image shape: {data_dict['image'].shape}")
print(f"image affine after Spacing:\n{data_dict['image_meta_dict']['affine']}")
```

```
image shape: (1, 224, 224, 232)
image affine after Spacing:
[[  0.  -1.   0. 223.]
 [ -1.   0.   0. 223.]
 [  0.   0.  -1. 231.]
 [  0.   0.   0.   1.]]
 ```
 
 ```python
 image = data_dict['image']
plt.figure("visualize", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[0, :, :, 0], cmap="gray")
plt.show()
```

> Random affine transformation

```python
rand_affine = RandAffined(
    keys=["image"],
    mode=("bilinear"),
    prob=1.0,
    spatial_size=(224, 224, 232),
    translate_range=(40, 40, 2),
    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
    scale_range=(0.15, 0.15, 0.15),
    padding_mode="border",
)
rand_affine.set_random_state(seed=123)
affined_data_dict = rand_affine(data_dict)
```

```python
print(f"image shape: {affined_data_dict['image'].shape}")  # image shape: (1, 224, 224, 232)
```

> Random elastic deformation

```python
rand_elastic = Rand3DElasticd(
    keys=["image"],
    mode=("bilinear"),
    prob=1.0,
    sigma_range=(5, 8),
    magnitude_range=(100, 200),
    spatial_size=(224, 224, 232),
    translate_range=(50, 50, 2),
    rotate_range=(np.pi / 36, np.pi / 36, np.pi),
    scale_range=(0.15, 0.15, 0.15),
    padding_mode="border",
)
rand_elastic.set_random_state(seed=123)
deformed_data_dict = rand_elastic(data_dict)
```

```python
print(f"image shape: {deformed_data_dict['image'].shape}")  # image shape: (1, 224, 224, 232)
```

size는 계속 그대로 유지.
