***Background***

The CIFAR-10 PyTorch dataset starts in an NHWC format, with RGB channel values ranging from 0 to 255.0. However, for optimal model performance, we want to permute the dataset to a NCHW format and normalize the data such that the mean value of channels is 0 and the standard deviation is 1
This project compares 3 methods of preprocessing the dataset to achieve this - running builtin PyTorch operations on the CPU, on the GPU, and using a custom CUDA kernel to do the preprocessing manually. It compares both execution speed and memory overhead. 

***Results***

The custom kernel resulted in significantly better preprocessing time, with an average time of 3.6ms per preprocess, whereas running PyTorch on the CUDA had an average time of 22.7ms per preprocess and the CPU had an average time of 94.2ms per preprocess. The custom kernel was thus around 5.6x faster than the default GPU and 25.7x faster than the default CPU. In addition, each preprocess for the custom kernel only used one kernel per pixel, whereas PyTorch CUDA allocated four kernels per pixel for operations, reducing the memory bandwidth.