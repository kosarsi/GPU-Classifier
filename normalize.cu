#include <cstdint>

__global__
void normalize_kernel(const uint8_t* data, float* output, int N, int H, int W, float mean0, float mean1, float mean2, float std0, float std1, float std2) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x; 

    int totalPixels = N * H * W;
    int imageSize = H * W; 

    // i is the pixel number
    for (int i = index; i < totalPixels; i += stride) {
        // Normalizes channels to have a mean of 0 and standard deviation of 1
        float channel0 = ((data[3 * i + 0] / 255.0f) - mean0) / std0;
        float channel1 = ((data[3 * i + 1] / 255.0f) - mean1) / std1;
        float channel2 = ((data[3 * i + 2] / 255.0f) - mean2) / std2;

        // Swaps format from NHWC to NCHW
        int pixelPosition = i % imageSize; 
        int imageIndex = i / imageSize; 
        int imageBase = imageIndex * 3 * imageSize; 

        output[imageBase + pixelPosition + imageSize * 0] = channel0;
        output[imageBase + pixelPosition + imageSize * 1] = channel1;
        output[imageBase + pixelPosition + imageSize * 2] = channel2;

    }
    
}

extern "C" void launch_kernel(const uint8_t* data, float* output, cudaStream_t stream, int N, int H, int W, float mean0, float mean1, float mean2, float std0, float std1, float std2) {
        
    normalize_kernel<<<64, 128, 0, stream>>>(data, output, N, H, W, mean0, mean1, mean2, std0, std1, std2);

}

