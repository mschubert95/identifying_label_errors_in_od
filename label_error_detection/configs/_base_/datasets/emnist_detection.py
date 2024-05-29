# dataset settings
dataset_type = 'EmnistDataset'
data_root = '/home/user/nutshell_emnist/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
data = dict(
    # samples_per_gpu=8,
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes=classes,
        # ann_file=data_root + 'train/annotations/emnistdetection20k_train_instances.json',
        ann_file=data_root + 'train/annotations/emnistdetection20k_train_instances.json',
        img_prefix=data_root + 'train/img/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        # ann_file=data_root + 'kitti_test.json',
        ann_file=data_root + 'val/annotations/emnistdetection_val_instances.json',
        img_prefix=data_root + 'val/img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        # ann_file=data_root + 'kitti_test.json',
        ann_file=data_root + 'test/annotations/emnistdetection_test_instances.json',
        img_prefix=data_root + 'test/img/',
        pipeline=test_pipeline),
    test_pert=dict(
        type='EmnistDataset',
        label_error_mode=True,
        classes=[
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ],
        ann_file=
        '/home/user/nutshell_emnist/test/annotations/emnistdetection_test_instances_20_switch_and_add.json',
        img_prefix='/home/user/nutshell_emnist/test/img/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(300, 300), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ])
        )
evaluation = dict(interval=1000, metric='bbox')
