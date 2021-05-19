_base_ = []
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook')

    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# model
label_convertor = dict(
    type='CTCConvertor', dict_file='data/textrecog/handwriting1/dict.txt', with_unknown=False, lower=False)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leakyRelu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)

train_cfg = None
test_cfg = None

# optimizer
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
total_epochs = 100

# data
img_norm_cfg = dict(mean=[0.5], std=[0.5])

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=1000,
        max_width=1000,
        keep_aspect_ratio=False),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=4,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'ori_shape', 'img_shape', 'valid_ratio']),
]

dataset_type = 'OCRDataset'

train_img_prefix = '/home/cuongnd/PycharmProjects/mmocr/data/textrecog/handwriting1'
train_ann_file = '/home/cuongnd/PycharmProjects/mmocr/data/textrecog/handwriting1/train.txt',
train = dict(
    type=dataset_type,
    img_prefix=train_img_prefix,
    ann_file=train_ann_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)


test_img_prefix = '/home/cuongnd/PycharmProjects/mmocr/data/textrecog/handwriting1'
test_ann_file = '/home/cuongnd/PycharmProjects/mmocr/data/textrecog/handwriting1/test.txt'

test = dict(
    type=dataset_type,
    img_prefix=test_img_prefix,
    ann_file=test_ann_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=test_pipeline,
    test_mode=True)

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=train,
    val= test,
    test=test)

evaluation = dict(interval=5, metric='acc')
cudnn_benchmark = True
