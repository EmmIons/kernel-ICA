# kernel-ICA
A python implementation for Bach F R, Jordan M I. Kernel independent component analysis[J]. Journal of machine learning research, 2002, 3(Jul): 1-48.
The kernel used here is **Gaussian kernel**, you can change to other kernel in `./lib/grammatrices.py.xxx_kernel_func()`.The model **Y=AX is realized by a Linear  network**, so the weights of the net is equal to W, which maps Y to X.
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
It will output the **Amari error** to compare the estimated W to the true W, as mentioned in the paper above
