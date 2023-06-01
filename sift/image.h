//
// Created by 18213 on 6/1/2023.
//

#ifndef SIFT_IMAGE_H
#define SIFT_IMAGE_H

#include <assert.h>
#include <cstdint>
#include <cstring>

namespace IMG {
    template <typename T>
    class Image
    {
    public:
        int cols;
        int rows;
        int step;
        int type;
        T *data;

        // Image type. Currently only support unsigned char and float.
        enum { TP_UCHAR, TP_FLOAT };

        // Padding each row to a multiple of 4 or 8 bytes
        enum { PAD_0, PAD_4, PAD_8 };

        Image() : cols(0), rows(0), step(0), type(TP_UCHAR), data(nullptr) {}

        // Create an image with specific size, padding, type and initial value.
        Image(int _rows, int _cols, int _padding = PAD_0, int _type = TP_UCHAR, T value = 0);

        // Copy construction function (Deep copy)
        Image(const Image<T> &img, int _padding = PAD_0, int _type = TP_UCHAR);

        ~Image() { release(); }

        // Create an image with specific size, padding and initial value.
        void create(int _rows, int _cols, int _padding = PAD_0, int _type = TP_UCHAR, T value = 0);

        // Release the image images.
        void release();

        // Clone the image (Deep copy)
        Image<T> clone(int _padding = PAD_0) const;

        // Copy assignment operator (Deep copy)
        Image<T> &operator=(const Image<T> &img);

        // Get the pointer to the specific row.
        T *operator[](int i) { return data + i * step; }

        // Subtraction operator
        Image<T> operator-(const Image<T> &img) const;

        // Downsample the image by 2x, nearest neighbor interpolation.
        Image<T> downsample_2x(int _padding = PAD_0) const;

        // Upsample the image by 2x, bilinear interpolation.
        Image<T> upsample_2x(int _padding = PAD_0) const;

        // Convert the image to unsigned char type.
        Image<uint8_t> to_uchar() const;

        // Convert the image to float type.
        Image<float> to_float() const;

    };

    template <typename T>
    Image<T>::Image(int _rows, int _cols, int _padding, int _type, T value)
    {
        create(_rows, _cols, _padding, _type, value);
    }

    template <typename T>
    Image<T>::Image(const Image<T> &img, int _padding, int _type)
    {
        create(img.rows, img.cols, _padding, _type);
        for (int i = 0; i < rows; i++) {
            memcpy(data + i * step, img.data + i * img.step, cols * sizeof(T));
        }
    }

    template <typename T>
    void Image<T>::create(int _rows, int _cols, int _padding, int _type, T value)
    {
        rows = _rows;
        cols = _cols;
        type = _type;
        switch (_padding) {
            case PAD_0:
                step = cols;
                break;
            case PAD_4:
                step = (cols + 3) & -4;
                break;
            case PAD_8:
                step = (cols + 7) & -8;
                break;
            default:
                step = cols;
                break;
        }
        data = new T[step * rows];
        for (int i = 0; i < rows; i++) {
            memset(data + i * step, value, cols * sizeof(T));
            memset(data + i * step + cols, 0, (step - cols) * sizeof(T));   // pad with 0
        }
    }

    template <typename T>
    void Image<T>::release()
    {
        if (data) {
            delete[] data;
            data = nullptr;
        }
        rows = 0;
        cols = 0;
        step = 0;
        type = TP_UCHAR;
    }

    template <typename T>
    Image<T> Image<T>::clone(int _padding) const
    {
        Image<T> img(rows, cols, _padding, type);
        for (int i = 0; i < rows; i++) {
            memcpy(img.data + i * img.step, data + i * step, cols * sizeof(T));
        }
        return img;
    }

    template <typename T>
    Image<T> &Image<T>::operator=(const Image<T> &img)
    {
        if (this == &img) {
            return *this;
        }
        if (rows != img.rows || cols != img.cols) {
            release();
            create(img.rows, img.step, PAD_0, img.type);    // use PAD_0 to avoid padding twice
            cols = img.cols;    // restore cols
        }
        for (int i = 0; i < rows; i++) {
            memcpy(data + i * step, img.data + i * img.step, cols * sizeof(T));
        }
        return *this;
    }

    template <typename T>
    Image<T> Image<T>::operator-(const Image<T> &img) const
    {
        assert(rows == img.rows && cols == img.cols);
        Image<T> res(rows, step, PAD_0, type); // keep the same padding with src1
        res.cols = cols;    // restore cols

        T *p_src1, *p_src2, *p_dst;
        for (int i = 0; i < rows; i++) {
            p_src1 = data + i * step;
            p_src2 = img.data + i * img.step;
            p_dst = res.data + i * res.step;
            for (int j = 0; j < cols; j++) {
                p_dst[j] = p_src1[j] - p_src2[j];
            }
        }
        return res;
    }

    template <typename T>
    Image<T> Image<T>::downsample_2x(int padding) const
    {
        Image<T> res(rows >> 1, cols >> 1, padding, type);
        T *p_src, *p_dst;
        for (int i = 0; i < res.rows; i++) {
            p_src = data + (i << 1) * step;
            p_dst = res.data + i * res.step;
            for (int j = 0; j < res.cols; ++j) {
                p_dst[j] = p_src[j << 1];
            }
        }
        return res;
    }

    template <typename T>
    Image<T> Image<T>::upsample_2x(int padding) const
    {
        Image<T> res(rows << 1, cols << 1, padding, type);
        T *src_data = data;
        T *dst_data = res.data;
        int r, c, idx;  // index of src_data
        float fr, fc, dr, dc;
        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < res.cols; ++j) {
                fr = i * 0.5f;
                fc = j * 0.5f;
                r = (int)fr;
                c = (int)fc;
                dr = fr - r;
                dc = fc - c;
                idx = r * step + c;
                // be careful of the boundary
                dst_data[i * res.step + j] = (1 - dr) * (1 - dc) * src_data[idx] +
                                             (1 - dr) * dc * (c < cols - 1 ? src_data[idx + 1] : src_data[idx]) +
                                             dr * (1 - dc) * (r < rows - 1 ? src_data[idx + step] : src_data[idx]) +
                                             dr * dc * ((c < cols - 1 && r < rows - 1) ? src_data[idx + step + 1] : src_data[idx]);
            }
        }
        return res;
    }

    template <typename T>
    Image<uint8_t> Image<T>::to_uchar() const
    {
        Image<uint8_t> res(rows, step, PAD_0, TP_UCHAR);
        res.cols = cols;    // restore cols

        T *p_src;
        uint8_t *p_dst;
        for (int i = 0; i < rows; i++) {
            p_src = data + i * step;
            p_dst = res.data + i * res.step;
            for (int j = 0; j < cols; ++j) {
                if (type == TP_FLOAT) {
                    if (p_src[j] < 0) {
                        p_dst[j] = 0;
                    } else if (p_src[j] > 255) {    // truncate
                        p_dst[j] = 255;
                    } else {
                        p_dst[j] = (uint8_t)p_src[j];
                    }
                } else {
                    p_dst[j] = p_src[j];
                }
            }
        }
        return res;
    }

    template <typename T>
    Image<float> Image<T>::to_float() const
    {
        Image<float> res(rows, step, PAD_0, TP_FLOAT);
        res.cols = cols;    // restore cols

        T *p_src;
        float *p_dst;
        for (int i = 0; i < rows; i++) {
            p_src = data + i * step;
            p_dst = res.data + i * res.step;
            for (int j = 0; j < cols; ++j) {
                p_dst[j] = (float)p_src[j];
            }
        }
        return res;
    }

} // IMG

#endif //SIFT_IMAGE_H
