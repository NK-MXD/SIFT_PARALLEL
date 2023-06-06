#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <omp.h>

#include "../sift.h"

void mtest_multithreadandSIMD() {
    SiftOmpandSIMD *sift_opencv_omp_simd;
    double t1, t2;
    double ratio = 0.6;
    int nthreads = omp_get_max_threads();
    std::cout<<nthreads<<std::endl;

    cv::Mat img0, img0_gray;
    std::vector<std::vector<cv::Mat>> gpyr0, dogpyr0;
    std::vector<cv::KeyPoint> kpts0;
    cv::Mat desc0;
    sift_opencv_omp_simd = new SiftOmpandSIMD(nthreads);
    img0 = cv::imread("/work/home/acmhsiv3ds/SIFT_PARALLEL-sift-parallel/sift/assets/peacock.jpg");
    cv::cvtColor(img0, img0_gray, cv::COLOR_BGR2GRAY);
    t1 = (double)cv::getTickCount();
    sift_opencv_omp_simd->detect(img0_gray, gpyr0, dogpyr0, kpts0);
    t1 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
    std::cout << "sift_opencv_omp_simd detect img0 time: " << t1 << " , kpts0 size: " << kpts0.size() << std::endl;

    t2 = (double)cv::getTickCount();
    sift_opencv_omp_simd->compute(gpyr0, kpts0, desc0);
    t2 = ((double)cv::getTickCount() - t2) / cv::getTickFrequency();
    std::cout << "sift_opencv_omp_simd compute img0 time: " << t2 << std::endl;
    std::cout << "sift_opencv_omp_simd sift total time: " << t1 + t2 << std::endl;

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
    /*cv::Mat img1, img2, img1_gray, img2_gray, img_match;
    std::vector<std::vector<cv::Mat>> gpyr1, dogpyr1, gpyr2, dogpyr2;
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good_matches;
    img1 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/sift/assets/test1.jpg");
    img2 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/sift/assets/test2.jpg");
    cv::resize(img1, img1, cv::Size(img1.cols / 2, img1.rows / 2));
    cv::resize(img2, img2, cv::Size(img2.cols / 2, img2.rows / 2));
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    
    t = (double)cv::getTickCount();
    sift_opencv_mt->detect(img1_gray, gpyr1, dogpyr1, kpts1);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift_opencv_mt detect img1 time: " << t << " , kpts1 size: " << kpts1.size() << std::endl;

    t = (double)cv::getTickCount();
    sift_opencv_mt->compute(gpyr1, kpts1, desc1);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift_opencv_mt compute img1 time: " << t << std::endl;

    t = (double)cv::getTickCount();
    sift_opencv_mt->detect(img2_gray, gpyr2, dogpyr2, kpts2);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift_opencv_mt detect img2 time: " << t << ", kpts2 size: " << kpts2.size() << std::endl;

    t = (double)cv::getTickCount();
    sift_opencv_mt->compute(gpyr2, kpts2, desc2);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift_opencv_mt compute img2 time: " << t << std::endl;

    matches.clear();
    matcher->knnMatch(desc1, desc2, matches, 2);
    std::cout << "sift_opencv_mt matches size: " << matches.size() << std::endl;

    good_matches.clear();
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < ratio * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
    std::cout << "sift_opencv_mt good matches size: " << good_matches.size() << std::endl;

    cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, img_match);

    cv::imshow("sift_opencv_mt", img_match);
    cv::waitKey(0);*/
}