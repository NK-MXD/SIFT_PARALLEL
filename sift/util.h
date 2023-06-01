//
// Created by 18213 on 6/1/2023.
//

#ifndef SIFT_UTIL_H
#define SIFT_UTIL_H

#include <opencv2/opencv.hpp>
#include "image.h"
#include "assert.h"
#include <iostream>

template <typename T>
cv::Mat img_to_mat(const IMG::Image<T> &img)
{
    assert(img.type == IMG::Image<T>::TP_FLOAT || img.type == IMG::Image<T>::TP_UCHAR);
    cv::Mat mat;
    if (img.type == IMG::Image<T>::TP_FLOAT) {
        mat = cv::Mat(img.rows, img.cols, CV_32FC1);
    } else {
        mat = cv::Mat(img.rows, img.cols, CV_8UC1);
    }
    const T *data = img.data;
    const T *p_img;
    T *p_mat;
    for (int i = 0; i < img.rows; ++i) {
        p_img = data + i * img.step;
        p_mat = mat.ptr<T>(i);
        for (int j = 0; j < img.cols; ++j) {
            p_mat[j] = p_img[j];
        }
    }
    return mat;
}

template <typename T>
IMG::Image<T> mat_to_img(const cv::Mat &mat, int type = IMG::Image<T>::TP_UCHAR, int padding = IMG::Image<T>::PAD_0)
{
    assert(mat.type() == CV_32FC1 || mat.type() == CV_8UC1);
    IMG::Image<T> img(mat.rows, mat.cols, padding, type);
    if (mat.type() == CV_32FC1) {
        const float *p_mat;
        T *p_img;
        for (int i = 0; i < img.rows; ++i) {
            p_mat = mat.ptr<float>(i);
            p_img = img.data + i * img.step;
            for (int j = 0; j < img.cols; ++j) {
                p_img[j] = (T)p_mat[j];
            }
        }
    } else {
        const uchar *p_mat;
        T *p_img;
        for (int i = 0; i < img.rows; ++i) {
            p_mat = mat.ptr<uchar>(i);
            p_img = img.data + i * img.step;
            for (int j = 0; j < img.cols; ++j) {
                p_img[j] = (T)p_mat[j];
            }
        }
    }
    return img;
}

#endif //SIFT_UTIL_H
