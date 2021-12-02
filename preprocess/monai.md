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
