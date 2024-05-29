# Identifying_Label_Errors_in_Object_Detection_Datasets_by_Loss_Inspection

This is supplementary to the submission Identifying_Label_Errors_in_Object_Detection_Datasets_by_Loss_Inspection.

We have implemented our label error detection framework in the open source [MMDetection tool box](https://github.com/open-mmlab/mmdetection). For testing, we suggest to clone the repository (tested for v2.20.0) and insert the folders 
```
    - configs
    - mmdet
```
into the repository structure (replacing some of the existing model definitions which is necessary to allow for, e.g., loss tracking during first and second stages).

## Utilization
To simulate label errors, start:
```bash
$ python3 ../transformation/simulate_label_errors.py
```
In order to start label error detection, utilize `tools/two_stage_approach.py` by passing a config file and a checkpoint path to it:
```bash
$ python3 ./tools/two_stage_approach.py <path/to/config.py> <path/to/checkpoint.py>
```
After that, the assignment of the label error proposals and the labels (including simulated label errors) is started with:
```bash
$ python3 ./tools/label_error_detection.py <path/to/config.py> <path/to/work_dir>
```

## Label Error Configs
The test config of the model needs an adjustment, to discriminate between standard detection and label error detection (see ./configs/swin/cascade_swin-t-p4-w7_fpn_1x_emnist_all.py):
```python3
test_cfg=dict(
    rpn=dict(
        nms_pre=1000,
        max_per_img=500,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0,
        metrics=False),         # is automatically set to True when doing label error detection
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
        metrics=False)          # is automatically set to True when doing label error detection
```
