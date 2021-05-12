import os

CONFIG_ROOT = os.path.dirname(__file__)

def full_path(sub_path, file=True):
    path = os.path.join(CONFIG_ROOT, sub_path)
    if not file and not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('full_path. Error makedirs', path)
    return path

visualize = False
config_path = full_path('weights/kie/sale_contract/sdmgr_unet16_60e_sale_contracts.py')
ckpt_path = full_path('weights/kie/sale_contract/epoch_20.pth')
viz_dir = full_path('viz/sale_contracts', file=False)
