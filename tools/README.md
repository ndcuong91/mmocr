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

### Finance invoices
Entities_list = ['VAT_amount', 'VAT_amount_val', 'VAT_rate', 'VAT_rate_val', 'account_no', 'address', 'address_val',
                 'amount_in_words', 'amount_in_words_val', 'bank', 'buyer', 'company_name', 'company_name_val', 'date',
                 'exchange_rate', 'exchange_rate_val', 'form', 'form_val', 'grand_total', 'grand_total_val', 'no',
                 'no_val', 'seller', 'serial', 'serial_val', 'tax_code', 'tax_code_val', 'total', 'total_val',
                 'website','other']

Ignore 'other'

| **Method** |  **train** |  **val** | **test** | ***input_size** | **Speed** | **Ckpt size** | **Note** |
| ------- | --------- | -------- | --------- | ---------- | ---------- | ---------- |---------- |
| PICK  | 0.   | 0.   | 0. | 480x960 | 3.3 fps | 890Mb | mEF 0.9019|
| SDMGR  | 0.9188   | 0.8828    |  0. | min: 512 max: 1024 | 10 fps | 20Mb | epoch 39|