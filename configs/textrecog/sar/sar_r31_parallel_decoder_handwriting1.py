_base_ = ['../../_base_/default_runtime.py']

checkpoint_config = dict(interval=5)

label_convertor = dict(
    type='AttnConvertor', dict_file='data/textrecog/handwriting1/dict.txt', with_unknown=False, lower=False)

model = dict(
    type='SARNet',
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(
        type='SAREncoder',
        enc_bi_rnn=False,
        enc_do_rnn=0.1,
        enc_gru=False,
    ),
    decoder=dict(
        type='ParallelSARDecoder',
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_do_rnn=0,
        dec_gru=False,
        pred_dropout=0.1,
        d_k=512,
        pred_concat=True),
    loss=dict(type='SARLoss'),
    label_convertor=label_convertor,
    max_seq_len=30)

# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 100

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=1000,
        keep_aspect_ratio=True),
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
    dict(type='LoadImageFromFile'),
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
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=train,
    val=test,
    test=test)

evaluation = dict(interval=5, metric='acc')
cudnn_benchmark = True