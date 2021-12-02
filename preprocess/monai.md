## MONAI 사용법

참고링크. [Load medical images](https://github.com/Project-MONAI/tutorials/blob/master/modules/load_medical_images.ipynb),
[3D image transforms](https://github.com/Project-MONAI/tutorials/blob/master/modules/3d_image_transforms.ipynb)

### LoadImage

이미지 불러오는 용도(NIfTI, DICOM, PNG format)

```python
loader = LoadImage(dtype=np.float32)
```

```python
image, metadata = loader(train_data_dicts[0]["image"])

print(f"input: {train_data_dicts[0]['image']}")
# input: /workspace/data/medical/Task09_Spleen/imagesTr/spleen_10.nii.gz

print(f"image shape: {image.shape}")
# image shape: (512, 512, 55)

print(f"image affine:\n{metadata['affine']}")
# image affine:
# [[   0.97656202    0.            0.         -499.02319336]
#  [   0.            0.97656202    0.         -499.02319336]
#  [   0.            0.            5.            0.        ]
#  [   0.            0.            0.            1.        ]]

print(f"image pixdim:\n{metadata['pixdim']}")
# image pixdim:
# [1.       0.976562 0.976562 5.       0.       0.       0.       0.      ]
```

결국 train_data_dict는 그냥 파일명이
