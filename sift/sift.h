#ifndef SIFT_SIFT_H
#define SIFT_SIFT_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/utility.hpp>

#include "omp.h"

const double SIFT_GAUSS_KERNEL_RATIO = 3;        //高斯核大小和标准差关系，size=2*(GAUSS_KERNEL_RATIO*sigma)+1,经常设置GAUSS_KERNEL_RATIO=2-3之间
const int SIFT_MAX_OCTAVES = 8;                  //金字塔最大组数
const float SIFT_CONTR_THR = 0.04f;              //默认是的对比度阈值(D(x))
const float SIFT_CURV_THR = 10.0f;               //关键点主曲率阈值
const float SIFT_INIT_SIGMA = 0.5f;              //输入图像的初始尺度
const int SIFT_IMG_BORDER = 2;                   //图像边界忽略的宽度
const int SIFT_MAX_INTERP_STEPS = 5;             //关键点精确插值次数
const int SIFT_ORI_HIST_BINS = 36;               //计算特征点方向直方图的BINS个数
const float SIFT_ORI_SIG_FCTR = 1.5f;            //计算特征点主方向时候，高斯窗口的标准差因子
const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;  //计算特征点主方向时，窗口半径因子
const float SIFT_ORI_PEAK_RATIO = 0.8f;          //计算特征点主方向时，直方图的峰值比
const int SIFT_DESCR_WIDTH = 4;                  //描述子直方图的网格大小(4x4)
const int SIFT_DESCR_HIST_BINS = 8;              //每个网格中直方图角度方向的维度
const float SIFT_DESCR_MAG_THR = 0.2f;           //描述子幅度阈值
const float SIFT_DESCR_SCL_FCTR = 3.0f;          //计算描述子时，每个网格的大小因子

typedef struct {
    int r;
    int c;
    int octave;
    int layer;
} LocalExtrema;

class Sift
{
protected:
    int nfeatures;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;
    int firstOctave;

public:
    Sift(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04,
           double edgeThreshold = 10.0, double sigma = 1.6, int firstOctave = -1) :
            nfeatures(nfeatures), nOctaveLayers(nOctaveLayers), contrastThreshold(contrastThreshold),
            edgeThreshold(edgeThreshold), sigma(sigma), firstOctave(firstOctave) {}

	//计算金字塔组数
	int num_octaves(const cv::Mat &img) const;

	//生成高斯金字塔第一组，第一层图像
	void create_init_img(const cv::Mat &image, cv::Mat &init) const;

	//创建高斯金字塔
	void build_gaussian_pyramid(const cv::Mat &init_img, std::vector<std::vector<cv::Mat>> &gpyr, int nOctaves) const;

	//创建高斯差分金字塔
	void build_dog_pyramid(std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<std::vector<cv::Mat>> &gpyr) const;

    // 计算图像的梯度幅值和方向
    float calc_orientation_hist(const cv::Mat &image, cv::Point pt, float scale, int n, float *hist) const;

    // 调整极值点
    bool adjust_local_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, cv::KeyPoint &kpt, int octave, int &layer, int &row, int &col, int nOctaveLayers, float contrastThreshold, float edgeThreshold, float sigma) const;

	//DOG金字塔特征点检测
	void find_scale_space_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<std::vector<cv::Mat>> &gpyr, std::vector<cv::KeyPoint> &kpts) const;

    //特征点检测
	void detect(const cv::Mat &image, std::vector<std::vector<cv::Mat>> &gpyr, std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint> &kpts) const;

    //计算sift local descriptor
    void calc_sift_descriptor(const cv::Mat &gauss_image, float main_angle, cv::Point2f pt, float scale, int d, int n, float *desc) const;

	//计算特征点的描述子
	void compute(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<cv::KeyPoint> &kpts, cv::Mat &desc) const;
};

class SiftOmp : public Sift
{
protected:
    int nthreads = omp_get_max_threads();

public:
    SiftOmp(int nthreads, int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04,
            double edgeThreshold = 10.0, double sigma = 1.6, int firstOctave = -1) :
            Sift(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, firstOctave), nthreads(nthreads) {}

    //创建高斯金字塔
    void build_gaussian_pyramid(const cv::Mat &init_img, std::vector<std::vector<cv::Mat>> &gpyr, int nOctaves) const;

    //创建高斯差分金字塔
    void build_dog_pyramid(std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<std::vector<cv::Mat>> &gpyr) const;

    // 计算图像的梯度幅值和方向
    float calc_orientation_hist(const cv::Mat &image, cv::Point pt, float scale, int n, float *hist) const;

    //DOG金字塔特征点检测
    void find_scale_space_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<std::vector<cv::Mat>> &gpyr, std::vector<cv::KeyPoint> &kpts) const;

    //特征点检测
    void detect(const cv::Mat &image, std::vector<std::vector<cv::Mat>> &gpyr, std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint> &kpts) const;

    //计算sift local descriptor
    void calc_sift_descriptor(const cv::Mat &gauss_image, float main_angle, cv::Point2f pt, float scale, int d, int n, float *desc) const;

    //计算特征点的描述子
    void compute(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<cv::KeyPoint> &kpts, cv::Mat &desc) const;

    void find_local_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<LocalExtrema> &extrema) const;

    void adjust(const std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<LocalExtrema> &extrema, std::vector<cv::KeyPoint> &kpts, std::vector<LocalExtrema> &extrema_adjust) const;

    void calc_orientation(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<cv::KeyPoint> &kpts, const std::vector<LocalExtrema> &extrema, std::vector<cv::KeyPoint> &kpts_res) const;
};

class SiftOpencvPara : public Sift
{
protected:
    int nthreads = cv::getNumThreads();

public:
    SiftOpencvPara(int nthreads, int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04,
            double edgeThreshold = 10.0, double sigma = 1.6, int firstOctave = -1) :
            Sift(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, firstOctave), nthreads(nthreads) {}

    //创建高斯金字塔
    void build_gaussian_pyramid(const cv::Mat &init_img, std::vector<std::vector<cv::Mat>> &gpyr, int nOctaves) const;

    //创建高斯差分金字塔
    void build_dog_pyramid(std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<std::vector<cv::Mat>> &gpyr) const;

    // 计算图像的梯度幅值和方向
    float calc_orientation_hist(const cv::Mat &image, cv::Point pt, float scale, int n, float *hist) const;

    //DOG金字塔特征点检测
    void find_scale_space_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<std::vector<cv::Mat>> &gpyr, std::vector<cv::KeyPoint> &kpts) const;

    //特征点检测
    void detect(const cv::Mat &image, std::vector<std::vector<cv::Mat>> &gpyr, std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint> &kpts) const;

    //计算sift local descriptor
    void calc_sift_descriptor(const cv::Mat &gauss_image, float main_angle, cv::Point2f pt, float scale, int d, int n, float *desc) const;

    //计算特征点的描述子
    void compute(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<cv::KeyPoint> &kpts, cv::Mat &desc) const;


    typedef struct
    {
        int r;
        int c;
        int octave;
        int layer;
    } LocalExtrema;
    void find_local_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<LocalExtrema> &extrema) const;

    void adjust(const std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<LocalExtrema> &extrema, std::vector<cv::KeyPoint> &kpts, std::vector<LocalExtrema> &extrema_adjust) const;

    void calc_orientation(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<cv::KeyPoint> &kpts, const std::vector<LocalExtrema> &extrema, std::vector<cv::KeyPoint> &kpts_res) const;
};



#endif //SIFT_SIFT_H
