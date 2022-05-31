from mmocr.utils.ocr import MMOCR

# Load models into memory
# ocr = MMOCR(det=None, recog_config='/home/cuongnd/PycharmProjects/mmocr/configs/textrecog/crnn/crnn_handwriting1.py',
#             recog_ckpt='/home/cuongnd/PycharmProjects/mmocr/work_dirs/textrecog/crnn/handwriting1/epoch_100.pth')

ocr = MMOCR(det = 'PANet_IC15',
            det_config='/home/cuongnd/PycharmProjects/mmocr/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py',
            recog ='SAR',
            recog_config = '/home/cuongnd/PycharmProjects/mmocr/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py')

img_dir = '/data_backup/cuongnd/gpdkkd/backtest/imgs/gpdkkd_v1.1/imgs/formal_002.jpg'
res_dir = '/home/cuongnd/PycharmProjects/mmocr/data/textrecog/handwriting1/test/AICR_test1/res'
# Inference
results = ocr.readtext(img_dir, output = res_dir, batch_mode=False, recog_batch_size=10, merge = True, imshow = False)


