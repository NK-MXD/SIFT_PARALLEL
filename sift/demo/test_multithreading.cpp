#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "../sift.h"

void test_omp() {
    Sift *sift;
    SiftOmp *sift_mt;
    double t;
    double ratio = 0.6;
    int nthreads = 8;

    cv::Mat img1, img2, img1_gray, img2_gray, img_match, img_match_mt;
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

    sift = new Sift();
    sift_mt = new SiftOmp(nthreads);

    t = (double)cv::getTickCount();
    sift->detect(img1_gray, gpyr1, dogpyr1, kpts1);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift detect img1 time: " << t << " , kpts1 size: " << kpts1.size() << std::endl;

    t = (double)cv::getTickCount();
    sift->compute(gpyr1, kpts1, desc1);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift compute img1 time: " << t << std::endl;

    t = (double)cv::getTickCount();
    sift->detect(img2_gray, gpyr2, dogpyr2, kpts2);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift detect img2 time: " << t << ", kpts2 size: " << kpts2.size() << std::endl;

    t = (double)cv::getTickCount();
    sift->compute(gpyr2, kpts2, desc2);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift compute img2 time: " << t << std::endl;

    matcher = cv::DescriptorMatcher::create("BruteForce");
    matcher->knnMatch(desc1, desc2, matches, 2);
    std::cout << "sift matches size: " << matches.size() << std::endl;

    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < ratio * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
    std::cout << "sift good matches size: " << good_matches.size() << std::endl;

    cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, img_match);

    t = (double)cv::getTickCount();
    sift_mt->detect(img1_gray, gpyr1, dogpyr1, kpts1);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift_mt detect img1 time: " << t << " , kpts1 size: " << kpts1.size() << std::endl;

    t = (double)cv::getTickCount();
    sift_mt->compute(gpyr1, kpts1, desc1);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift_mt compute img1 time: " << t << std::endl;

    t = (double)cv::getTickCount();
    sift_mt->detect(img2_gray, gpyr2, dogpyr2, kpts2);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift_mt detect img2 time: " << t << ", kpts2 size: " << kpts2.size() << std::endl;

    t = (double)cv::getTickCount();
    sift_mt->compute(gpyr2, kpts2, desc2);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "sift_mt compute img2 time: " << t << std::endl;

    matches.clear();
    matcher->knnMatch(desc1, desc2, matches, 2);
    std::cout << "sift_mt matches size: " << matches.size() << std::endl;

    good_matches.clear();
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < ratio * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
    std::cout << "sift_mt good matches size: " << good_matches.size() << std::endl;

    cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, img_match_mt);

    cv::imshow("sift", img_match);
    cv::imshow("sift_mt", img_match_mt);
    cv::waitKey(0);
}