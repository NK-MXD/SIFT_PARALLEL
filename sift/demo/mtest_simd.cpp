#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <omp.h>

#include "../sift.h"

void mtest_simd(){
    Sift *sift_simd;
    double t1, t2;
    double ratio = 0.6;

    cv::Mat img0, img0_gray;
    std::vector<std::vector<cv::Mat>> gpyr0, dogpyr0;
    std::vector<cv::KeyPoint> kpts0;
    cv::Mat desc0;
    sift_simd = new Sift();
    img0 = cv::imread("/work/home/acmhsiv3ds/SIFT_PARALLEL-sift-parallel/sift/assets/peacock.jpg");
    cv::cvtColor(img0, img0_gray, cv::COLOR_BGR2GRAY);
    t1 = (double)cv::getTickCount();
    sift_simd->detect(img0_gray, gpyr0, dogpyr0, kpts0);
    t1 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
    std::cout << "sift_simd detect img0 time: " << t1 << "s, kpts0 size: " << kpts0.size() << std::endl;

    t2 = (double)cv::getTickCount();
    sift_simd->compute(gpyr0, kpts0, desc0);
    t2 = ((double)cv::getTickCount() - t2) / cv::getTickFrequency();
    std::cout << "sift_simd compute img0 time: " << t2 << "s" << std::endl;
    std::cout << "sift_simd sift total time: " << t1 + t2 << "s" << std::endl;

    // 标准测试
    // 标准sift生成的图像:
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

    // 检测关键点和计算描述符
    std::vector<cv::KeyPoint> keypoints00;
    cv::Mat descriptors00;
    double t0 = (double)cv::getTickCount();
    detector->detect(img0, keypoints00, cv::noArray());
    t0 = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    std::cout << "Standard sift detect time: " << t0 << "s, kpts0 size: " << keypoints00.size() << std::endl;
    double t01 = (double)cv::getTickCount();
    detector->compute(img0, keypoints00, descriptors00);
    // detector->detectAndCompute(img0, cv::noArray(), keypoints00, descriptors00);
    t01 = ((double)cv::getTickCount() - t01) / cv::getTickFrequency();
    std::cout << "Standard sift compute time: " << t01 << "s" << std::endl;
    std::cout << "Standard sift total time: " << (t0 + t01) << "s" << std::endl;
    std::cout << "Standard sift / Our SIFT: " << std::fixed << std::setprecision(3) << (t0 + t01) / (t1 + t2) << std::endl;
}