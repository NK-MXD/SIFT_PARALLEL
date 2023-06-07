#include <iostream>
#include <cassert>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "sift_cuda.cuh"
#include "util_cuda.cuh"

int main() {
    int devNum = 0;
    cv::Mat img1, img2;
    cv::Mat img1_u8, img2_u8;
    img1_u8 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/sift-cuda/assets/peacock.jpg", 0);
    img2_u8 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/sift-cuda/assets/peacock.jpg", 0);
    img1_u8.convertTo(img1, CV_32FC1);
    img2_u8.convertTo(img2, CV_32FC1);
    assert(img1.data && img2.data);
    assert(deviceInit(devNum));

    double t = 0;
    t = cv::getTickCount();

    CudaImage img1_cuda, img2_cuda;
    img1_cuda.Allocate(img1.cols, img1.rows, iAlignUp(img1.cols, 128), false, nullptr, (float *) img1.data);
    img2_cuda.Allocate(img2.cols, img2.rows, iAlignUp(img2.cols, 128), false, nullptr, (float *) img2.data);
    img1_cuda.Download();
    img2_cuda.Download();

    SiftData siftData1, siftData2;
    float initBlur = 0.f;
    float thresh = 3.f;
    InitSiftData(siftData1, 65536, true, true);
    InitSiftData(siftData2, 65536, true, true);

    float *memoryTmp = AllocSiftTempMemory(img1_cuda.width, img1_cuda.height, 5, false);
    ExtractSift(siftData1, img1_cuda, 5, initBlur, thresh, 0.0f, false, memoryTmp);
    FreeSiftTempMemory(memoryTmp);

    memoryTmp = AllocSiftTempMemory(img2_cuda.width, img2_cuda.height, 5, false);
    ExtractSift(siftData2, img2_cuda, 5, initBlur, thresh, 0.0f, false, memoryTmp);
    FreeSiftTempMemory(memoryTmp);

    t = double ((cv::getTickCount()) - t);
    std::cout << "Extraction time = " << t / cv::getTickFrequency() * 1000. << " ms" << std::endl;


//    matchAndDraw(siftData1, siftData2, img1_u8, img2_u8);

    FreeSiftData(siftData1);
    FreeSiftData(siftData2);

    return 0;
}
