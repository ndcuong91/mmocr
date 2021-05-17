img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_scale = 1024
min_scale = 512
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='KIEFormatBundle'),
    dict(type='Collect', keys=['img', 'relations', 'texts', 'gt_bboxes'])
]
dataset_type = 'KIEDataset'
data_root = '/home/duycuong/home_data/mmocr/kie/finance_invoices'
loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='LineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations']))
train = dict(
    type='KIEDataset',
    ann_file='/home/duycuong/home_data/mmocr/kie/finance_invoices/train.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='KIEFormatBundle'),
        dict(
            type='Collect',
            keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'])
    ],
    img_prefix='/home/duycuong/home_data/mmocr/kie/finance_invoices',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    dict_file='/home/duycuong/home_data/mmocr/kie/finance_invoices/dict.txt',
    test_mode=False)
val = dict(
    type='KIEDataset',
    ann_file='/home/duycuong/home_data/mmocr/kie/finance_invoices/val.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='KIEFormatBundle'),
        dict(type='Collect', keys=['img', 'relations', 'texts', 'gt_bboxes'])
    ],
    img_prefix='/home/duycuong/home_data/mmocr/kie/finance_invoices',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    dict_file='/home/duycuong/home_data/mmocr/kie/finance_invoices/dict.txt',
    test_mode=True)
test = dict(
    type='KIEDataset',
    ann_file='/home/duycuong/home_data/mmocr/kie/finance_invoices/test.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='KIEFormatBundle'),
        dict(type='Collect', keys=['img', 'relations', 'texts', 'gt_bboxes'])
    ],
    img_prefix='/home/duycuong/home_data/mmocr/kie/finance_invoices',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    dict_file='/home/duycuong/home_data/mmocr/kie/finance_invoices/dict.txt',
    test_mode=True)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type='KIEDataset',
        ann_file=
        '/home/duycuong/home_data/mmocr/kie/finance_invoices/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='KIEFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'])
        ],
        img_prefix='/home/duycuong/home_data/mmocr/kie/finance_invoices',
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])),
        dict_file=
        '/home/duycuong/home_data/mmocr/kie/finance_invoices/dict.txt',
        test_mode=False),
    val=dict(
        type='KIEDataset',
        ann_file='/home/duycuong/home_data/mmocr/kie/finance_invoices/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='KIEFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'relations', 'texts', 'gt_bboxes'])
        ],
        img_prefix='/home/duycuong/home_data/mmocr/kie/finance_invoices',
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])),
        dict_file=
        '/home/duycuong/home_data/mmocr/kie/finance_invoices/dict.txt',
        test_mode=True),
    test=dict(
        type='KIEDataset',
        ann_file='/home/duycuong/home_data/mmocr/kie/finance_invoices/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='KIEFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'relations', 'texts', 'gt_bboxes'])
        ],
        img_prefix='/home/duycuong/home_data/mmocr/kie/finance_invoices',
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])),
        dict_file=
        '/home/duycuong/home_data/mmocr/kie/finance_invoices/dict.txt',
        test_mode=True))
evaluation = dict(
    interval=1,
    metric='macro_f1',
    metric_options=dict(macro_f1=dict(ignores=[0, 31])))
model = dict(
    type='SDMGR',
    backbone=dict(type='UNet', base_channels=16),
    bbox_head=dict(
        type='SDMGRHead', visual_dim=16, num_chars=240, num_classes=32),
    visual_modality=True,
    train_cfg=None,
    test_cfg=None,
    class_list=
    '/home/duycuong/home_data/mmocr/kie/finance_invoices/class_list.txt')
optimizer = dict(type='Adam', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1,
    step=[40, 50])
total_epochs = 60
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
work_dir = './work_dirs/kie/finance_invoices/'
gpu_ids = range(0, 1)
