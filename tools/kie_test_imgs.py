import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv import Config
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.models import build_detector

from sklearn.metrics import f1_score

config_path = '../work_dirs/kie/finance_invoices/20210427_182401/sdmgr_unet16_60e_finance_invoices.py'
checkpoint_path = '../work_dirs/kie/finance_invoices/20210427_182401/epoch_39.pth'
save_dir_path = '../viz/finance_invoices'


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMOCR visualize for kie model.')
    parser.add_argument('--config', help='Test config file path.', default=config_path)
    parser.add_argument('--checkpoint', help='Checkpoint file.', default=checkpoint_path)
    parser.add_argument('--show', action='store_true', help='Show results.')
    parser.add_argument(
        '--show-dir', help='Directory where the output images will be saved.', default=save_dir_path)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def test(model, data_loader, show=False, out_dir=None, eval=True):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    y_pred = []
    y_gt = []
    for idx, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            img_tensor = data['img'].data[0]
            img_metas = data['img_metas'].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            gt_bboxes = [data['gt_bboxes'].data[0][0].numpy().tolist()]

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                box_ann_infos = dataset.data_infos[idx]['annotations']
                node_gt = [box_ann_info['label'] for box_ann_info in box_ann_infos]

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                vis_img, node_pred = model.module.show_result(
                    img_show,
                    result[i],
                    gt_bboxes[i],
                    show=show,
                    out_file=out_file)
                if len(node_pred) != len(node_gt):
                    print('Here')
                y_pred.extend(node_pred)
                y_gt.extend(node_gt)

        for _ in range(batch_size):
            prog_bar.update()

    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
              30]
    if eval:
        print('\nF1 scores of each class.................................................')
        count = 0
        total_F1 = 0
        for label in labels:
            score = eval_marco_F1(y_pred=y_pred,
                                  y_gt=y_gt,
                                  labels=[label],
                                  )
            print(str(label).ljust(20), score)
            if score > -10:
                count += 1
                total_F1 += score
        print('average F1 in', count, 'class:', total_F1 / count)


        # score = eval_marco_F1(y_pred=y_pred,
        #                       y_gt=y_gt,
        #                       # labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        #                       labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15, 16, 17, 18, 19, 20,21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        # print('\nScore:', score)

    return results



def eval_marco_F1(y_pred, y_gt, labels=None):
    '''

    :param y_pred:
    :param y_gt:
    :param labels:
    :return:
    '''
    return f1_score(y_gt, y_pred, labels=labels, average='macro', zero_division=0)


def main():
    args = parse_args()
    assert args.show or args.show_dir, ('Please specify at least one '
                                        'operation (show the results / save )'
                                        'the results with the argument '
                                        '"--show" or "--show-dir".')

    cfg = Config.fromfile(args.config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])
    import time
    begin = time.time()
    test(model, data_loader, args.show, args.show_dir)
    end = time.time()
    print('\ninference times', end - begin, 'seconds')


if __name__ == '__main__':
    main()
