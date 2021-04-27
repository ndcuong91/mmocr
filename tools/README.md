## KIE models comparison 
Metric: Marco F1
### Sale contract
Entities_list = ['contract_no','exporter_name','exporter_add','importer_name','payment_method',
                    'ben_bank_name','ben_add','ben_name','ben_acc','swift_code','other']

Ignore 'other'

| **Method** |  **train** |  **val** | **test** | ***input_size** | **Speed** | **Ckpt size** | **Note** |
| ------- | --------- | -------- | --------- | ---------- | ---------- | ---------- |---------- |
| PICK  | 0.8501   | 0.7961   | 0.7117 | 768x1088 | 3.3 fps | 890Mb | mEF 0.9019|
| SDMGR  | 0.9859   | 0.975    |  0.7809 | min: 512 max: 1024 | 10 fps | 20Mb | epoch 43|