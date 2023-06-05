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

    // img1 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/sift/assets/test1.jpg");
    // img2 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/sift/assets/test2.jpg");
    img1 = cv::imread("/work/home/acmhsiv3ds/SIFT_PARALLEL-sift-parallel/sift/assets/peacock.jpg");
    // if(!img1.data){
    //     std::cout<<"empty()";
    //     return;
    // }else{
    //     std::cout<<img1.size();
    // }
    // cv::resize(img1, img1, cv::Size(img1.cols / 2, img1.rows / 2));
    // cv::resize(img2, img2, cv::Size(img2.cols / 2, img2.rows / 2));
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    sift = new Sift();
    double processtime0 = (double)cv::getTickCount();
    sift->detect(img1_gray, gpyr1, dogpyr1, kpts1);
    double processtimedes0 = (double)cv::getTickCount();
    sift->compute(gpyr1, kpts1, desc1);
    processtimedes0 = ((double)cv::getTickCount() - processtimedes0) / cv::getTickFrequency();
    processtime0 = ((double)cv::getTickCount() - processtime0) / cv::getTickFrequency();
    std::cout << "ExtractDescriptor算法执行时间： " << processtimedes0 << "s" << std::endl;
    std::cout << "算法执行总时间： " << processtime0 << "s" << std::endl;
    std::cout << "特征点个数： " << kpts1.size() << std::endl;
    // sift->detect(img2_gray, gpyr2, dogpyr2, kpts2);
    // sift->compute(gpyr2, kpts2, desc2);

    // matcher = cv::DescriptorMatcher::create("BruteForce");
    // matcher->knnMatch(desc1, desc2, matches, 2);

    // for (int i = 0; i < matches.size(); i++) {
    //     if (matches[i][0].distance < 0.6 * matches[i][1].distance) {
    //         good_matches.push_back(matches[i][0]);
    //     }
    // }

    // cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, img_match);
    // cv::imshow("match", img_match);
    // cv::waitKey(0);

    // 标准测试
    // 标准sift生成的图像:
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

    // 检测关键点和计算描述符
    std::vector<cv::KeyPoint> keypoints00;
    cv::Mat descriptors00;
    double processtime00 = (double)cv::getTickCount();
    detector->detectAndCompute(img1, cv::noArray(), keypoints00, descriptors00);
    processtime00 = ((double)cv::getTickCount() - processtime00) / cv::getTickFrequency();
    std::cout << "标准sift算法执行总时间： " << processtime00 << "s" << std::endl;
    std::cout << "Our SIFT/Standard SIFT： " << std::fixed << std::setprecision(3) << processtime00 / processtime0 << std::endl;
    // cv::Mat output_image00;
    // cv::drawKeypoints(image0, keypoints00, output_image00, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

}