//
// Created by 18213 on 6/1/2023.
//
#include "../../image.h"
#include "../../util.h"
#include <iostream>
#include <cstdint>

int main() {
    cv::Mat mat = cv::imread("C:\\Users\\18213\\Documents\\mypro\\SIFT_PARALLEL\\sift\\images\\peacock.jpg", cv::IMREAD_GRAYSCALE);
    IMG::Image<uint8_t> image = mat_to_img<uint8_t>(mat);
    cv::Mat mat2 = img_to_mat<uint8_t>(image);
    cv::imshow("image", mat2);
    cv::waitKey(0);
    return 0;
}
