#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "../sift.h"

void test_match() {
    Sift *sift;
    cv::Mat img1, img2, img1_gray, img2_gray, img_match;
    std::vector<std::vector<cv::Mat>> gpyr1, dogpyr1, gpyr2, dogpyr2;
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good_matches;
    double t;

    img1 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/sift/assets/peacock.jpg");
    img2 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/sift/assets/peacock.jpg");
//    cv::resize(img1, img1, cv::Size(img1.cols / 2, img1.rows / 2));
//    cv::resize(img2, img2, cv::Size(img2.cols / 2, img2.rows / 2));
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    sift = new Sift();

    t = (double)cv::getTickCount();
    sift->detect(img1_gray, gpyr1, dogpyr1, kpts1);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "detect time: " << t << "s" << std::endl;

    t = (double)cv::getTickCount();
    sift->compute(gpyr1, kpts1, desc1);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "compute time: " << t << "s" << std::endl;

    sift->detect(img2_gray, gpyr2, dogpyr2, kpts2);
    sift->compute(gpyr2, kpts2, desc2);

    matcher = cv::DescriptorMatcher::create("BruteForce");

    t = (double)cv::getTickCount();
    matcher->knnMatch(desc1, desc2, matches, 2);
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < 0.6 * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "match time: " << t << "s" << std::endl;

    cv::Mat img_kpts;
    cv::drawKeypoints(img1, kpts1, img_kpts);
//    cv::imshow("kpts", img_kpts);
////    cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, img_match);
////    cv::imshow("match", img_match);
//    cv::waitKey(0);
    cv::imwrite("/home/guo/mypro/SIFT_PARALLEL/sift/build/peacock.jpg", img_kpts);
}