_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_pipelines/satrn_pipeline.py'
]

# train_list = {{_base_.train_list}}
# test_list = {{_base_.test_list}}
#
# train_pipeline = {{_base_.train_pipeline}}
# test_pipeline = {{_base_.test_pipeline}}

label_convertor = dict(
    type='AttnConvertor',
    dict_list=list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz| 0123456789ĂÂÊÔƠƯÁẮẤÉẾÍÓỐỚÚỨÝÀẰẦÈỀÌÒỒỜÙỪỲẢẲẨĐẺỂỈỎỔỞỦỬỶÃẴẪẼỄĨÕỖỠŨỮỸẠẶẬẸỆỊỌỘỢỤỰỴăâêôơưáắấéếíóốớúứýàằầèềìòồờùừỳảẳẩđẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ\'*:,@.-(#%")/~!^&_+={}[]\;<>?※”$€£¥₫°²™ā–'), with_unknown=False, lower=False)

model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN', input_channels=3, hidden_dim=512),
    encoder=dict(
        type='SatrnEncoder',
        n_layers=12,
        n_head=8,
        d_k=512 // 8,
        d_v=512 // 8,
        d_model=512,
        n_position=100,
        d_inner=512 * 4,
        dropout=0.1),
    decoder=dict(
        type='NRTRDecoder',
        n_layers=6,
        d_embedding=512,
        n_head=8,
        d_model=512,
        d_inner=512 * 4,
        d_k=512 // 8,
        d_v=512 // 8),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=25)

# optimizer
optimizer = dict(type='Adam', lr=3e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 6

# data
img_norm_cfg = dict(mean=[0.5], std=[0.5])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=100,
        keep_aspect_ratio=False),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio','resize_shape'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=100,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'ori_shape', 'img_shape', 'valid_ratio','resize_shape']),
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
            separator='\t')),
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
            separator='\t')),
    pipeline=test_pipeline,
    test_mode=True)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=train,
    val= test,
    test=test)

cudnn_benchmark = True
evaluation = dict(interval=1, metric='acc')
