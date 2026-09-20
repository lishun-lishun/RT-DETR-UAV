# Archived upstream DCNv4 extension

The current `rtdetr_r18vd_dut_anti_uav_p2_dcnv4.yml` experiment does **not**
import or require this native extension. Its self-contained UAV-DCNv4 operator
is implemented in `src/nn/backbone/plugin_points/dcnv4.py` and uses the
torchvision installation already required by RT-DETR.

This directory only preserves upstream source code for reference and
reproducibility of the superseded native-wrapper experiment. Do not compile or
install it for the current P2 UAV-DCNv4 experiment.
