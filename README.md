# kernel-ICA
A python implementation for Bach F R, Jordan M I. Kernel independent component analysis[J]. Journal of machine learning research, 2002, 3(Jul): 1-48.
The kernel used here is **Gaussian kernel**, you can change to your kernel in `./lib/grammatrices.py`
## Requirements
`python==3.10.10`
`scipy==1.10.1`
`numpy==1.24.2`
`torch==1.13.1`

## Train
`python main.py`
It will save the true W and estimated W in `./weights`

## Evaluate
`python Amari_error.py`
It will output the **Amari error** to evaluate the estimated W, based on the paper mentioned above
