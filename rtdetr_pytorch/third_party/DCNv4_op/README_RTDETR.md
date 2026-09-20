# RT-DETR DCNv4 extension

This directory is the DCNv4 CUDA extension required by the
`rtdetr_r18vd_dut_anti_uav_p2_dcnv4.yml` experiment. It is bundled here so the
RT-DETR training project no longer depends on the separate reference project.

From the `rtdetr_pytorch` directory on the Linux training server, activate the
same Python environment used for training and run:

```bash
python -m pip install ninja
cd third_party/DCNv4_op
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 python -m pip install -v . --no-build-isolation
cd ../..
```

Verify the native extension before training:

```bash
python -c "import DCNv4.ext as ext; print(hasattr(ext, 'dcnv4_forward'), hasattr(ext, 'dcnv4_backward'))"
```

The expected output is `True True`. The CUDA toolkit reported by `nvcc
--version` should match `torch.version.cuda`; the CUDA version displayed by
`nvidia-smi` is the driver capability and is not necessarily the toolkit used
to build PyTorch.
