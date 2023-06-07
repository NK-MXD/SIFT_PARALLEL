#ifndef SIFT_CUDA_SIFT_CUDA_CUH
#define SIFT_CUDA_SIFT_CUDA_CUH

#include <opencv2/opencv.hpp>

#include "img_cuda.cuh"
#include "util_cuda.cuh"

#define NUM_SCALES      5
#define SCALEDOWN_W    64 // 60     // Scale down thread block width
#define SCALEDOWN_H    16 // 8      // Scale down thread block height
#define SCALEUP_W      64           // Scale up thread block width
#define SCALEUP_H       8           // Scale up thread block height
#define MINMAX_W       30 //32      // Find point thread block width
#define MINMAX_H        8 //16      // Find point thread block height
#define LAPLACE_W     128 // 56     // Laplace thread block width
#define LAPLACE_H       4           // Laplace rows per thread
#define LAPLACE_S   (NUM_SCALES+3)  // Number of laplace scales
#define LAPLACE_R       4           // Laplace filter kernel radius
#define LOWPASS_W      24 //56
#define LOWPASS_H      32 //16
#define LOWPASS_R       4

typedef struct {
    float xpos;
    float ypos;
    float scale;
    float sharpness;
    float edgeness;
    float orientation;
    float score;
    float ambiguity;
    int match;
    float match_xpos;
    float match_ypos;
    float match_error;
    float subsampling;
    float empty[3];
    float data[128];
} SiftPoint;

typedef struct {
    int numPts;
    int maxPts;
    SiftPoint *h_data;  // host
    SiftPoint *d_data;  // device
} SiftData;

float *AllocSiftTempMemory(int width, int height, int nOctaves, bool scaleUp = false);
void FreeSiftTempMemory(float *memoryTmp);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);
void PrintSiftData(SiftData &data);

double ExtractSift(SiftData &siftData, CudaImage &img, int nOctave, double initBlur, float thresh, float lowestScale = 0.0f, bool scaleUp = false, float *tempMemory = nullptr);
double ExtractSiftLoop(SiftData &siftData, CudaImage &img, int nOctave, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp);

double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave);
double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);
//double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);

void matchAndDraw(SiftData &siftData1, SiftData &siftData2, cv::Mat &img1, cv::Mat &img2);
//double MatchSiftData(SiftData &data1, SiftData &data2);
//double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);


double ScaleDown(CudaImage &res, CudaImage &src, float variance);
double ScaleUp(CudaImage &res, CudaImage &src);
double RescalePositions(SiftData &siftData, float scale);
double LowPass(CudaImage &res, CudaImage &src, float scale);
void PrepareLaplaceKernels(int nOctaves, float initBlur, float *kernel);


#endif //SIFT_CUDA_SIFT_CUDA_CUH
