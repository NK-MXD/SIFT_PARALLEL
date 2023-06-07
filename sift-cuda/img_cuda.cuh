#ifndef SIFT_CUDA_IMG_CUDA_CUH
#define SIFT_CUDA_IMG_CUDA_CUH

#include <cstdlib>
#include <cstdio>
#include "util_cuda.cuh"

static int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
static int iDivDown(int a, int b) { return a / b; }
static int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }
static int iAlignDown(int a, int b) { return a - a % b; }

class CudaImage {
public:
    int width, height;
    int pitch{};
    float *h_data;
    float *d_data;
    float *t_data;
    bool d_internalAlloc;
    bool h_internalAlloc;

    CudaImage() : width(0), height(0), d_data(nullptr), h_data(nullptr), t_data(nullptr), d_internalAlloc(false), h_internalAlloc(false) {}

    ~CudaImage() {
        if (d_internalAlloc && d_data != nullptr)
            safeCall(cudaFree(d_data));
        d_data = nullptr;
        if (h_internalAlloc && h_data != nullptr)
            free(h_data);
        h_data = nullptr;
        if (t_data != nullptr)
            safeCall(cudaFreeArray((cudaArray *) t_data));
        t_data = nullptr;
    }

    void Allocate(int _width, int _height, int _pitch, bool withHost, float *devMem = nullptr, float *hostMem = nullptr) {
        width = _width;
        height = _height;
        pitch = _pitch;
        d_data = devMem;
        h_data = hostMem;
        t_data = nullptr;
        if (!devMem) {
            safeCall(cudaMallocPitch((void **) &d_data, (size_t *) &pitch, (size_t) (sizeof(float) * width), (size_t) height));
            pitch /= sizeof(float);
            if (d_data == nullptr)
                printf("Failed to allocate device data\n");
            d_internalAlloc = true;
        }
        if (withHost && !hostMem) {
            h_data = (float *) malloc(sizeof(float) * pitch * height);
            h_internalAlloc = true;
        }
    }

    [[nodiscard]] double Download() const {
        TimerGPU timer(nullptr);
        size_t p = sizeof(float) * pitch;
        if (d_data && h_data)
            safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice));
        return timer.read();
    }

    [[nodiscard]] double Readback() const {
        TimerGPU timer(nullptr);
        size_t p = sizeof(float) * pitch;
        if (d_data && h_data)
            safeCall(cudaMemcpy2D(h_data, sizeof(float) * width, d_data, p, sizeof(float) * width, height, cudaMemcpyDeviceToHost));
        return timer.read();
    }

    double InitTexture() {
        TimerGPU timer(nullptr);
        cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>();
        safeCall(cudaMallocArray((cudaArray **) &t_data, &t_desc, pitch, height));
        if (!t_data) {
            printf("Error allocating array data\n");
            return 0.0;
        }
//        if (d_data != nullptr && t_data != nullptr)
//            safeCall(cudaMemcpy2DToArray((cudaArray *) t_data, 0, 0, d_data, sizeof(float) * pitch, sizeof(float) * pitch, height, cudaMemcpyDeviceToDevice));
        return timer.read();
    }

    double CopyToTexture(CudaImage &dst, bool host) {
        TimerGPU timer(nullptr);
        if (host) {
            if (dst.t_data != nullptr && h_data != nullptr)
                safeCall(cudaMemcpy2DToArray((cudaArray *) dst.t_data, 0, 0, h_data, sizeof(float) * pitch, sizeof(float) * pitch, height, cudaMemcpyHostToDevice));
        } else {
            if (dst.t_data != nullptr && d_data != nullptr)
                safeCall(cudaMemcpy2DToArray((cudaArray *) dst.t_data, 0, 0, d_data, sizeof(float) * pitch, sizeof(float) * pitch, height, cudaMemcpyDeviceToDevice));
        }
        return timer.read();
    }
};

#endif //SIFT_CUDA_IMG_CUDA_CUH
