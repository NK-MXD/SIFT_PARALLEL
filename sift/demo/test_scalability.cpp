#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <omp.h>

#include "../sift.h"

void test_scalability() {
    cv::Mat img_ori = cv::imread("/home/guo/mypro/SIFT_PARALLEL/sift/assets/peacock.jpg", 0);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            // resize to 1/2, 1/4, 1/8, 1/1
            cv::Mat img;
            cv::resize(img_ori, img, cv::Size(img_ori.cols / (1 << i), img_ori.rows / (1 << i)));
            int nthreads = (1 << j);
            double t;

            std::cout << img.rows << std::endl << img.cols << std::endl;
            std::cout << nthreads << std::endl;

            std::vector<std::vector<cv::Mat>> gpyr, dogpyr;
            std::vector<cv::KeyPoint> kpts;
            cv::Mat desc;
            Sift *sift = new Sift();    // serial
            cv::Ptr<cv::SIFT> sift_cv = cv::SIFT::create();    // opencv sift
            SiftOmp *sift_mt = new SiftOmp(nthreads);    // parallel omp
            SiftOmpandSIMD *sift_mt_simd = new SiftOmpandSIMD(nthreads);    // parallel omp and simd

            kpts.clear();
            t = (double)cv::getTickCount();
            sift->detect(img, gpyr, dogpyr, kpts);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            std::cout << t << std::endl;

            t = (double)cv::getTickCount();
            sift->compute(gpyr, kpts, desc);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            std::cout << t << std::endl;
            std::cout << kpts.size() << std::endl;

            kpts.clear();
            t = (double)cv::getTickCount();
            sift_cv->detect(img, kpts);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            std::cout << t << std::endl;

            t = (double)cv::getTickCount();
            sift_cv->compute(img, kpts, desc);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            std::cout << t << std::endl;
            std::cout << kpts.size() << std::endl;

            kpts.clear();
            t = (double)cv::getTickCount();
            sift_mt->detect(img, gpyr, dogpyr, kpts);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            std::cout << t << std::endl;

            t = (double)cv::getTickCount();
            sift_mt->compute(gpyr, kpts, desc);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            std::cout << t << std::endl;
            std::cout << kpts.size() << std::endl;

            kpts.clear();
            t = (double)cv::getTickCount();
            sift_mt_simd->detect(img, gpyr, dogpyr, kpts);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            std::cout << t << std::endl;

            t = (double)cv::getTickCount();
            sift_mt_simd->compute(gpyr, kpts, desc);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            std::cout << t << std::endl;
            std::cout << kpts.size() << std::endl;
        }
    }

    
}