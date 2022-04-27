from mmocr.utils.ocr import MMOCR

# Load models into memory
ocr = MMOCR(det=None, recog_config='/home/cuongnd/PycharmProjects/mmocr/configs/textrecog/crnn/crnn_handwriting1.py',
            recog_ckpt='/home/cuongnd/PycharmProjects/mmocr/work_dirs/textrecog/crnn/handwriting1/epoch_100.pth')

img_dir = '/home/cuongnd/PycharmProjects/mmocr/data/textrecog/handwriting1/test/AICR_test1/AICR_P0000005/0005_1.jpg'
res_dir = '/home/cuongnd/PycharmProjects/mmocr/data/textrecog/handwriting1/test/AICR_test1/res'
# Inference
results = ocr.readtext(img_dir, output = res_dir, batch_mode=False, single_batch_size = 10)
