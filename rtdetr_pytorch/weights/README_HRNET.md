# HRNetV2-W18 ImageNet checkpoint

For an offline training server, place this file in the current directory:

```text
hrnetv2_w18-8cb57bb9.pth
```

Expected full path from the project root:

```text
weights/hrnetv2_w18-8cb57bb9.pth
```

Source URL:

```text
https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w18-8cb57bb9.pth
```

The model checks this project-local path before trying Torch Hub. In a
three-rank launch, only global rank 0 is allowed to download; the other ranks
wait and then read the shared cache. Loading remains strict for every backbone
key, while classification-head-only keys are ignored.
