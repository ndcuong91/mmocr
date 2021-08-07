# Benchmark

Compare  speed between algorithm

## Training speed

### Text recognition

GPU: GTX 1080 ti

| trainset | input_size (h x w)  | batch_size | fps | Time for 1 epoch with 1M samples  |cudnn_benchmark|
| :------: | :----------: | :--------:  | :--------: | :---: | :---: |
|  CRNN  |   32 x 1000    |  64   | 200     | ~ 86 min | True |
|  CRNN  |   32 x 1000    |  64   | 137      | ~ 126 min | False |
|  SAR paralllel  |   32 x 1000    |  16   | 28.7     | ~ 600 min | True |

