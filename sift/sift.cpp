#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/core/types.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>

#include "sift.h"
#include "mylist.h"

int Sift::num_octaves(const cv::Mat &img) const 
{
    int n;
    float min_size;
    
    min_size = (float)std::min(img.rows, img.cols);
    n = cvRound(std::log(min_size) / std::log(2.0) - 2) - firstOctave;
    if (n > SIFT_MAX_OCTAVES)
        n = SIFT_MAX_OCTAVES;

    return n;
}

void Sift::create_init_img(const cv::Mat &img, cv::Mat &init) const
{
    cv::Mat gray, f_img;
    double sig_diff, scale;
    int ksize;

    if (img.channels() != 1)
        cvtColor(img, gray, cv::COLOR_RGB2GRAY);
    else
        gray = img.clone();
    gray.convertTo(f_img, CV_32FC1, 1.0 / 255.0, 0);
    if (firstOctave < 0) {
        cv::resize(f_img, f_img, cv::Size(2 * f_img.cols, 2 * f_img.rows), 0, 0, cv::INTER_LINEAR);
        scale = 2.0;
    } else {
        scale = 1.0;
    }
    sig_diff = sqrt(sigma * sigma - (scale * SIFT_INIT_SIGMA) * (scale * SIFT_INIT_SIGMA));
    ksize = 2 * cvRound(SIFT_GAUSS_KERNEL_RATIO * sig_diff) + 1;
    cv::GaussianBlur(f_img, init, cv::Size(ksize, ksize), sig_diff, sig_diff);
}

void Sift::build_gaussian_pyramid(const cv::Mat &init_img, std::vector<std::vector<cv::Mat>> &gpyr, int nOctaves) const {
    const int nLayers = nOctaveLayers + 3;
    std::vector<double> sig(nLayers);
    gpyr.resize(nOctaves);
    for (int i = 0; i < nOctaves; i++)
        gpyr[i].resize(nLayers);

    sig[0] = sigma;
    double k = pow(2.0, 1.0 / nOctaveLayers);
    double prev, curr;
    for (int i = 1; i < nLayers; i++) {
        prev = pow(k, double(i - 1)) * sigma;
        curr = k * prev;
        sig[i] = sqrt(curr * curr - prev * prev);
    }

    for (int i = 0; i < nOctaves; i++)
    {
        for (int j = 0; j < nLayers; j++)
        {
            if (i == 0 && j == 0) {
                gpyr[0][0] = init_img;
            } else if (j == 0) {
                cv::Mat &prev = gpyr[i - 1][nOctaveLayers];
                cv::resize(prev, gpyr[i][0], cv::Size(prev.cols >> 1, prev.rows >> 1), 0, 0, cv::INTER_NEAREST);
            } else {
                int ksize = 2 * cvRound(SIFT_GAUSS_KERNEL_RATIO * sig[j]) + 1;
                cv::GaussianBlur(gpyr[i][j - 1], gpyr[i][j], cv::Size(ksize, ksize), sig[j], sig[j]);
            }
        }
    }
}

void Sift::build_dog_pyramid(std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<std::vector<cv::Mat>> &gpyr) const
{
    int nOctaves = gpyr.size();
    int nLayers = nOctaveLayers + 2;
    dogpyr.resize(nOctaves);
    for (int i = 0; i < nOctaves; ++i)
        dogpyr[i].resize(nLayers);

	for (int i = 0; i < nOctaves; i++)
		for (int j = 0; j < nLayers; j++)
             cv::subtract(gpyr[i][j + 1], gpyr[i][j], dogpyr[i][j], cv::noArray(), CV_32FC1);
}

float Sift::calc_orientation_hist(const cv::Mat &image, cv::Point pt, float scale, int n, float *hist) const
{
	int radius = cvRound(SIFT_ORI_RADIUS * scale);  //特征点邻域半径(3*1.5*scale)
	int len = (2 * radius + 1) * (2 * radius + 1);       //特征点邻域像素总个数（最大值）
	float sigma = SIFT_ORI_RADIUS*scale;                 //特征点邻域高斯权重标准差(1.5*scale)
	float exp_scale = -1.f / (2 * sigma * sigma);
	cv::AutoBuffer<float> buffer(5 * len + n + 4);  //使用AutoBuffer分配一段内存，这里多出4个空间的目的是为了方便后面平滑直方图的需要
	float *X = buffer, *Y = X + len, *Mag = Y + len, *Ori = Mag + len, *W = Ori + len;  	//X保存水平差分，Y保存数值差分，Mag保存梯度幅度，Ori保存梯度角度，W保存高斯权重
	float *temp_hist = W + len + 2;                     //临时保存直方图数据
	for (int i = 0; i < n; i++)
		temp_hist[i] = 0.f;//数据清零

    //计算邻域像素的水平差分和竖直差分
    int k = 0;  // local counter for points within radius
    int x, y;
    float dx, dy;
    for (int i = -radius; i <= radius; ++i)
    {
        y = pt.y + i;
        if (y <= 0 || y >= image.rows - 1)
            continue;
        for (int j = -radius; j < radius; ++j) {
            x = pt.x + j;
            if (x <= 0 || x >= image.cols - 1)
                continue;
            dx = image.at<float>(y, x + 1) - image.at<float>(y, x - 1);
            dy = image.at<float>(y + 1, x) - image.at<float>(y - 1, x);
            X[k] = dx;
            Y[k] = dy;
            W[k] = (i * i + j * j) * exp_scale;
            ++k;
        }
    }

    //计算邻域像素的梯度幅度,梯度方向，高斯权重
    len = k;
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    // 计算直方图
    for (int i = 0; i < len; i++) {
        int bin = cvRound((n / 360.f) * Ori[i]);    //bin的范围约束在[0,(n-1)]
        if (bin >= n)
            bin -= n;
        if (bin < 0)
            bin += n;
        temp_hist[bin] += Mag[i] * W[i];
    }

    //平滑直方图
    temp_hist[-1] = temp_hist[n - 1];
    temp_hist[-2] = temp_hist[n - 2];
    temp_hist[n] = temp_hist[0];
    temp_hist[n + 1] = temp_hist[1];
    for (int i = 0; i < n; i++)
        hist[i] = (temp_hist[i - 2] + temp_hist[i + 2]) * 0.0625f +
                  (temp_hist[i - 1] + temp_hist[i + 1]) * 0.25f +
                  temp_hist[i] * 0.375f;
    float max_value = hist[0];
    for (int i = 1; i < n; ++i)
        if (hist[i] > max_value)
            max_value = hist[i];

    return max_value;
}

bool Sift::adjust_local_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, cv::KeyPoint &kpt, int octave, int &layer,
                                 int &row, int &col, int nOctaveLayers, float contrastThreshold, float edgeThreshold, float sigma) const
{
    float xi = 0, xr = 0, xc = 0;
    int i = 0;
    for (; i < SIFT_MAX_INTERP_STEPS; ++i)//最大迭代次数
    {
        const cv::Mat &img = dogpyr[octave][layer];//当前层图像索引
        const cv::Mat &prev = dogpyr[octave][layer - 1];//之前层图像索引
        const cv::Mat &next = dogpyr[octave][layer + 1];//下一层图像索引

        //特征点位置x方向，y方向,尺度方向的一阶偏导数
        float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * (1.f / 2.f);
        float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * (1.f / 2.f);
        float dz = (next.at<float>(row, col) - prev.at<float>(row, col)) * (1.f / 2.f);

        //计算特征点位置二阶偏导数
        float v2 = img.at<float>(row, col);
        float dxx = img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
        float dyy = img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
        float dzz = prev.at<float>(row, col) + next.at<float>(row, col) - 2 * v2;

        //计算特征点周围混合二阶偏导数
        float dxy = (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
                     img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) * (1.f / 4.f);
        float dxz = (next.at<float>(row, col + 1) + prev.at<float>(row, col - 1) -
                     next.at<float>(row, col - 1) - prev.at<float>(row, col + 1)) * (1.f / 4.f);
        float dyz = (next.at<float>(row + 1, col) + prev.at<float>(row - 1, col) -
                     next.at<float>(row - 1, col) - prev.at<float>(row + 1, col)) * (1.f / 4.f);

        cv::Matx33f H(dxx, dxy, dxz,
                      dxy, dyy, dyz,
                      dxz, dyz, dzz);

        cv::Vec3f dD(dx, dy, dz);

        // 用Hessian矩阵的逆矩阵乘以梯度向量，得到偏移量
        cv::Vec3f X = H.solve(dD, cv::DECOMP_SVD);

        xc = -X[0];//x方向偏移量
        xr = -X[1];//y方向偏移量
        xi = -X[2];//尺度方向偏移量

        //如果三个方向偏移量都小于0.5，说明已经找到特征点准确位置
        if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xi) < 0.5f)
            break;

        //如果其中一个方向的偏移量过大，则删除该点
        if (abs(xc) > (float) (INT_MAX / 3) ||
            abs(xr) > (float) (INT_MAX / 3) ||
            abs(xi) > (float) (INT_MAX / 3))
            return false;

        col += cvRound(xc);
        row += cvRound(xr);
        layer += cvRound(xi);

        //如果特征点定位在边界区域，同样也需要删除
        if (layer < 1 || layer > nOctaveLayers ||
            col < SIFT_IMG_BORDER || col > img.cols - SIFT_IMG_BORDER ||
            row < SIFT_IMG_BORDER || row > img.rows - SIFT_IMG_BORDER)
            return false;
    }

    //如果i=MAX_INTERP_STEPS，说明循环结束也没有满足条件，即该特征点需要被删除
    if (i >= SIFT_MAX_INTERP_STEPS)
        return false;

    // 再次删除低响应点
    //再次计算特征点位置x方向，y方向,尺度方向的一阶偏导数
    {
        const cv::Mat &img = dogpyr[octave][layer];
        const cv::Mat &prev = dogpyr[octave][layer - 1];
        const cv::Mat &next = dogpyr[octave][layer + 1];

        float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * 0.5f;
        float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * 0.5f;
        float dz = (next.at<float>(row, col) - prev.at<float>(row, col)) * 0.5f;
        cv::Matx31f dD(dx, dy, dz);
        float t = dD.dot(cv::Matx31f(xc, xr, xi));

        float contr = img.at<float>(row, col) + t * 0.5f;//特征点响应
        if (abs(contr) < contrastThreshold / (float) nOctaveLayers) //Low建议contr阈值是0.03，但是RobHess等建议阈值为0.04/nOctaveLayers
            return false;

        // 删除边缘响应比较强的点
        //再次计算特征点位置二阶偏导数
        float v2 = img.at<float>(row, col);
        float dxx = img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
        float dyy = img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
        float dxy = (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
                     img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) * (1.f / 4.f);
        float det = dxx * dyy - dxy * dxy;
        float trace = dxx + dyy;
        if (det < 0 || (trace * trace * edgeThreshold >= det * (edgeThreshold + 1) * (edgeThreshold + 1)))
            return false;

        // 保存该特征点信息
        kpt.pt.x = ((float) col + xc) * (float) (1 << octave);  //相对于最底层的图像的x坐标
        kpt.pt.y = ((float) row + xr) * (float) (1 << octave);  //相对于最底层图像的y坐标
        kpt.octave = octave + (layer << 8);                     //组号保存在低字节，层号保存在高字节
        kpt.size = sigma * powf(2.f, ((float) layer + xi) / (float) nOctaveLayers) * (float) (1 << octave);  //相对于最底层图像的尺度
        kpt.response = abs(contr);      //特征点响应值
        return true;
    }
}

void Sift::find_scale_space_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<std::vector<cv::Mat>> &gpyr,
                                    std::vector<cv::KeyPoint> &kpts) const
{
    kpts.clear();
	int nOctaves = (int)dogpyr.size();
	float threshold = (float)(contrastThreshold / nOctaveLayers);   //Low文章建议threshold是0.03，Rob Hess等人使用0.04/nOctaveLayers作为阈值
	const int n = SIFT_ORI_HIST_BINS;   //n=36
	float hist[n];
	cv::KeyPoint kpt;

    // DEBUG
//    int extrema_cnt = 0;
    int extrema_adjust_cnt = 0;

	for (int i = 0; i < nOctaves; ++i)//对于每一组
	{
		for (int j = 1; j <= nOctaveLayers; ++j)//对于组内每一层
		{
			const cv::Mat &curr_img = dogpyr[i][j];//当前层
			const cv::Mat &prev_img = dogpyr[i][j - 1];//之前层
			const cv::Mat &next_img = dogpyr[i][j + 1];
			int num_row = curr_img.rows;
			int num_col = curr_img.cols;//获得当前组图像的大小
			size_t step = curr_img.step1();//一行元素所占宽度

            for (int r = SIFT_IMG_BORDER; r < num_row - SIFT_IMG_BORDER; ++r)
            {
                const float* curr_ptr = curr_img.ptr<float>(r);
                const float* prev_ptr = prev_img.ptr<float>(r);
                const float* next_ptr = next_img.ptr<float>(r);

                for (int c = SIFT_IMG_BORDER; c < num_col - SIFT_IMG_BORDER; ++c)
                {
                    float val = curr_ptr[c];
					bool is_extrema = false;
					is_extrema = (abs(val) > threshold && ((val > 0 && val >= curr_ptr[c - 1] && val >= curr_ptr[c + 1] &&
                            val >= curr_ptr[c - step - 1] && val >= curr_ptr[c - step] && val >= curr_ptr[c - step + 1] &&
                            val >= curr_ptr[c + step - 1] && val >= curr_ptr[c + step] && val >= curr_ptr[c + step + 1] &&

                            val >= prev_ptr[c] && val >= prev_ptr[c - 1] && val >= prev_ptr[c + 1] &&
                            val >= prev_ptr[c - step - 1] && val >= prev_ptr[c - step] && val >= prev_ptr[c - step + 1] &&
                            val >= prev_ptr[c + step - 1] && val >= prev_ptr[c + step] && val >= prev_ptr[c + step + 1] &&

                            val >= next_ptr[c] && val >= next_ptr[c - 1] && val >= next_ptr[c + 1] &&
                            val >= next_ptr[c - step - 1] && val >= next_ptr[c - step] && val >= next_ptr[c - step + 1] &&
                            val >= next_ptr[c + step - 1] && val >= next_ptr[c + step] && val >= next_ptr[c + step + 1])
                            ||
                            (val < 0 && val <= curr_ptr[c - 1] && val <= curr_ptr[c + 1] &&
							val <= curr_ptr[c - step - 1] && val <= curr_ptr[c - step] && val <= curr_ptr[c - step + 1] &&
							val <= curr_ptr[c + step - 1] && val <= curr_ptr[c + step] && val <= curr_ptr[c + step + 1] &&

							val <= prev_ptr[c] && val <= prev_ptr[c - 1] && val <= prev_ptr[c + 1] &&
							val <= prev_ptr[c - step - 1] && val <= prev_ptr[c - step] && val <= prev_ptr[c - step + 1] &&
							val <= prev_ptr[c + step - 1] && val <= prev_ptr[c + step] && val <= prev_ptr[c + step + 1] &&

							val <= next_ptr[c] && val <= next_ptr[c - 1] && val <= next_ptr[c + 1] &&
							val <= next_ptr[c - step - 1] && val <= next_ptr[c - step] && val <= next_ptr[c - step + 1] &&
							val <= next_ptr[c + step - 1] && val <= next_ptr[c + step] && val <= next_ptr[c + step + 1])));
					if (is_extrema) {
						//++numKeys;
						//获得特征点初始行号，列号，组号，组内层号
						int r1 = r, c1 = c, octave = i, layer = j;
                        // DEBUG
//                        extrema_cnt++;
						if (!adjust_local_extrema(dogpyr, kpt, octave, layer, r1, c1,
                                                  nOctaveLayers, (float)contrastThreshold,
                                                  (float)edgeThreshold, (float)sigma))
						{
							continue;//如果该初始点不满足条件，则不保存改点
						}
                        extrema_adjust_cnt++;

						float scale = kpt.size / float (1 << octave);//该特征点相对于本组的尺度
						float max_hist = calc_orientation_hist(gpyr[octave][layer],
                                                               cv::Point(c1, r1), scale, n, hist);
						float mag_thr = max_hist * SIFT_ORI_PEAK_RATIO;

						for (int i = 0; i < n; ++i)
						{
							int left=0, right=0;
							if (i == 0)
								left = n - 1;
							else
								left = i - 1;

							if (i == n - 1)
								right = 0;
							else
								right = i + 1;

							if (hist[i] > hist[left] && hist[i] > hist[right] && hist[i] >= mag_thr)
							{
								float bin = i + 0.5f*(hist[left] - hist[right]) / (hist[left] + hist[right] - 2 * hist[i]);
								if (bin < 0)
									bin = bin + n;
								if (bin >= n)
									bin = bin - n;

								kpt.angle = (360.f / n)*bin;//特征点的主方向0-360度
								kpts.push_back(kpt);//保存该特征点
							}
						}
					}
				}
			}
		}
	}

    // DEBUG
//    std::cout << "sift find " << extrema_cnt << " local extrema" << std::endl;  // result: turns out that bugs remains at `adjust`
//    std::cout << "after adjust, remains " << extrema_adjust_cnt << " local extrema" << std::endl; // result: turns out that bugs not to do with `adjust`
}

void Sift::calc_sift_descriptor(const cv::Mat &gauss_image, float main_angle, cv::Point2f pt, float scale, int d, int n, float *descriptor) const
{
    cv::Point ptxy(cvRound(pt.x), cvRound(pt.y));//坐标取整
    float cos_t = cosf(-main_angle * (float) (CV_PI / 180));
    float sin_t = sinf(-main_angle * (float) (CV_PI / 180));
    float bins_per_rad = n / 360.f;//n=8
    float exp_scale = -1.f / (d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scale;//每个网格的宽度
    int radius = cvRound(hist_width * (d + 1) * sqrt(2) * 0.5f);//特征点邻域半径

    int rows = gauss_image.rows, cols = gauss_image.cols;
    radius = std::min(radius, (int) sqrt((double) rows * rows + cols * cols));
    cos_t = cos_t / hist_width;
    sin_t = sin_t / hist_width;

    int len = (2 * radius + 1) * (2 * radius + 1);
    int histlen = (d + 2) * (d + 2) * (n + 2);

    cv::AutoBuffer<float> buf(7 * len + histlen);
    //X保存水平差分，Y保存竖直差分，Mag保存梯度幅度，Angle保存特征点方向,W保存高斯权重
    float *X = buf, *Y = X + len, *Mag = Y + len, *Angle = Mag + len, *W = Angle + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

    //首先清空直方图数据
    for (int i = 0; i < d + 2; ++i) {
        for (int j = 0; j < d + 2; ++j) {
            for (int k = 0; k < n + 2; ++k)
                hist[(i * (d + 2) + j) * (n + 2) + k] = 0.f;
        }
    }

    //计算特征点邻域范围内每个像素的差分核高斯权重的指数部分
    int k = 0;
    for (int i = -radius; i < radius; ++i) {
        for (int j = -radius; j < radius; ++j) {
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d / 2 - 0.5f;
            float cbin = c_rot + d / 2 - 0.5f;
            int r = ptxy.y + i, c = ptxy.x + j;

            //这里rbin,cbin范围是(-1,d)
            if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
                float dx = gauss_image.at<float>(r, c + 1) - gauss_image.at<float>(r, c - 1);
                float dy = gauss_image.at<float>(r + 1, c) - gauss_image.at<float>(r - 1, c);
                X[k] = dx; //水平差分
                Y[k] = dy;//竖直差分
                RBin[k] = rbin;
                CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale;//高斯权值的指数部分
                ++k;
            }
        }
    }

    //计算像素梯度幅度，梯度角度，和高斯权值
    len = k;
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Angle, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    //计算每个特征点的描述子
    for (k = 0; k < len; ++k) {
        float rbin = RBin[k], cbin = CBin[k];//rbin,cbin范围是(-1,d)
        float obin = (Angle[k] - main_angle) * bins_per_rad;
        float mag = Mag[k] * W[k];

        int r0 = cvFloor(rbin);//ro取值集合是{-1,0,1,2，3}
        int c0 = cvFloor(cbin);//c0取值集合是{-1，0，1，2，3}
        int o0 = cvFloor(obin);
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        //限制范围为[0,n)
        if (o0 < 0)
            o0 += n;
        if (o0 >= n)
            o0 -= n;

        //使用三线性插值方法，计算直方图
        float v_r1 = mag * rbin;//第二行分配的值
        float v_r0 = mag - v_r1;//第一行分配的值

        float v_rc11 = v_r1 * cbin;
        float v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0 * cbin;
        float v_rc00 = v_r0 - v_rc01;

        float v_rco111 = v_rc11 * obin;
        float v_rco110 = v_rc11 - v_rco111;

        float v_rco101 = v_rc10 * obin;
        float v_rco100 = v_rc10 - v_rco101;

        float v_rco011 = v_rc01 * obin;
        float v_rco010 = v_rc01 - v_rco011;

        float v_rco001 = v_rc00 * obin;
        float v_rco000 = v_rc00 - v_rco001;

        //该像素所在网格的索引
        int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
        hist[idx] += v_rco000;
        hist[idx + 1] += v_rco001;
        hist[idx + n + 2] += v_rco010;
        hist[idx + n + 3] += v_rco011;
        hist[idx + (d + 2) * (n + 2)] += v_rco100;
        hist[idx + (d + 2) * (n + 2) + 1] += v_rco101;
        hist[idx + (d + 3) * (n + 2)] += v_rco110;
        hist[idx + (d + 3) * (n + 2) + 1] += v_rco111;
    }

    //由于圆周循环的特性，对计算以后幅角小于 0 度或大于 360 度的值重新进行调整，使
    //其在 0～360 度之间
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
            hist[idx] += hist[idx + n];
            for (k = 0; k < n; ++k)
                descriptor[(i * d + j) * n + k] = hist[idx + k];
        }
    }

    //对描述子进行归一化
    int lenght = d * d * n;
    float norm = 0;
    for (int i = 0; i < lenght; ++i) {
        norm += descriptor[i] * descriptor[i];
    }
    norm = sqrt(norm);
    for (int i = 0; i < lenght; ++i) {
        descriptor[i] /= norm;
    }

    //阈值截断
    for (int i = 0; i < lenght; ++i) {
        descriptor[i] = std::min(descriptor[i], SIFT_DESCR_MAG_THR);
    }

    //再次归一化
    norm = 0;
    for (int i = 0; i < lenght; ++i) {
        norm += descriptor[i] * descriptor[i];
    }
    norm = sqrt(norm);
    for (int i = 0; i < lenght; ++i) {
        descriptor[i] /= norm;
    }
}


void Sift::compute(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) const
{
	int d = SIFT_DESCR_WIDTH;//d=4,特征点邻域网格个数是d x d
	int n = SIFT_DESCR_HIST_BINS;//n=8,每个网格特征点梯度角度等分为8个方向
    int kpts_num = (int)keypoints.size();
	descriptors.create(kpts_num, d*d*n, CV_32FC1);//分配空间

    std::vector<cv::KeyPoint> kpts;
    if (firstOctave < 0) {  // keypoints中的特征点位置是相对于原图片而言的，因此这里需要将特征点的位置扩大2倍
        kpts.resize(kpts_num);
        for (size_t i = 0; i < kpts_num; ++i) {
            kpts[i].pt = keypoints[i].pt * 2.f;
            kpts[i].octave = keypoints[i].octave;
            kpts[i].size = keypoints[i].size * 2.f;
            kpts[i].angle = keypoints[i].angle;
        }
    } else {
        kpts = keypoints;   // shallow copy
    }
	for (size_t i = 0; i < kpts_num; ++i)//对于每一个特征点
	{
		int octaves, layer;
		//得到特征点所在的组号，层号
		octaves = kpts[i].octave & 255;
		layer = (kpts[i].octave >> 8) & 255;

		//得到特征点相对于本组的坐标，不是最底层
		cv::Point2f pt(kpts[i].pt.x / (float)(1 << octaves), kpts[i].pt.y / (float)(1 << octaves));
		float scale = kpts[i].size / (float)(1 << octaves);//得到特征点相对于本组的尺度
		float main_angle = kpts[i].angle;//特征点主方向

		//计算该点的描述子
		calc_sift_descriptor(gpyr[octaves][layer], main_angle, pt, scale, d, n, descriptors.ptr<float>((int)i));
	}
}

void Sift::detect(const cv::Mat &image, std::vector<std::vector<cv::Mat>> &gpyr, std::vector<std::vector<cv::Mat>> &dogpyr,
                  std::vector<cv::KeyPoint> &kpts) const
{
	if (image.empty() || image.depth() != CV_8U)
		CV_Error(cv::Error::StsBadArg,"输入图像为空，或者图像深度不是CV_8U");

	//计算高斯金字塔组数
	int nOctaves = num_octaves(image);
    //计时
    double counttime = 0;

	//生成高斯金字塔第一层图像
	cv::Mat init_gauss;
    counttime = (double)cv::getTickCount();
	create_init_img(image, init_gauss);
    counttime = ((double)cv::getTickCount() - counttime) / cv::getTickFrequency();
    std::cout << "create_init_img 算法执行时间： " << counttime << "s" << std::endl;

	//生成高斯尺度空间图像
    counttime = (double)cv::getTickCount();
	build_gaussian_pyramid(init_gauss, gpyr, nOctaves);
    counttime = ((double)cv::getTickCount() - counttime) / cv::getTickFrequency();
    std::cout << "build_gaussian_pyramid 算法执行时间： " << counttime << "s" << std::endl;

	//生成高斯差分金字塔(DOG金字塔，or LOG金字塔)
    counttime = (double)cv::getTickCount();
	build_dog_pyramid(dogpyr, gpyr);
    counttime = ((double)cv::getTickCount() - counttime) / cv::getTickFrequency();
    std::cout << "build_dog_pyramid 算法执行时间： " << counttime << "s" << std::endl;

	//在DOG金字塔上检测特征点
    counttime = (double)cv::getTickCount();
	find_scale_space_extrema(dogpyr, gpyr, kpts);
    counttime = ((double)cv::getTickCount() - counttime) / cv::getTickFrequency();
    std::cout << "find_scale_space_extrema 算法执行时间： " << counttime << "s" << std::endl;

	if (nfeatures != 0 && nfeatures < (int)kpts.size())
	{
		sort(kpts.begin(), kpts.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b)
		{
            return a.response>b.response;
        });
		kpts.erase(kpts.begin() + nfeatures, kpts.end());
	}
    if (kpts.size() == 0)
        std::cout << "No keypoints are detected!" << std::endl;

    for (auto &kpt : kpts) {        // 重新调整特征点的坐标，若初始图像经过上采样，则特征点坐标是相对于上采样图像的，因此调整回相对于原图像
        kpt.pt /= 2.f;
    }
}

void SiftOmp::build_gaussian_pyramid(const cv::Mat &init_img, std::vector<std::vector<cv::Mat>> &gpyr,
                                     int nOctaves) const {
    const int nLayers = nOctaveLayers + 3;
    std::vector<double> sig(nLayers);
    gpyr.resize(nOctaves);
    for (int i = 0; i < nOctaves; i++)
        gpyr[i].resize(nLayers);

    sig[0] = sigma;
    double k = pow(2.0, 1.0 / nOctaveLayers);

#pragma omp parallel for num_threads(nthreads) schedule(static) default(none) shared(nLayers, k, sigma, sig)
    for (int i = 1; i < nLayers; i++) {
        double prev = pow(k, double(i - 1)) * sigma;
        double curr = k * prev;
        sig[i] = sqrt(curr * curr - prev * prev);
    }

    // 该层循环存在数据依赖问题，并行需要进一步研究，可否采用直接求不相邻高斯图像的方法？
    // TODO
    for (int i = 0; i < nOctaves; i++) {
        for (int j = 0; j < nLayers; j++) {
            if (i == 0 && j == 0) {
                gpyr[0][0] = init_img;
            } else if (j == 0) {
                cv::Mat &prev = gpyr[i - 1][nOctaveLayers];
                cv::resize(prev, gpyr[i][0], cv::Size(prev.cols >> 1, prev.rows >> 1), 0, 0, cv::INTER_NEAREST);
            } else {
                int ksize = 2 * cvRound(SIFT_GAUSS_KERNEL_RATIO * sig[j]) + 1;
                cv::GaussianBlur(gpyr[i][j - 1], gpyr[i][j], cv::Size(ksize, ksize), sig[j], sig[j]);
            }
        }
    }
}

void SiftOmp::build_dog_pyramid(std::vector<std::vector<cv::Mat>> &dogpyr,
                                const std::vector<std::vector<cv::Mat>> &gpyr) const {
    int nOctaves = gpyr.size();
    int nLayers = nOctaveLayers + 2;
    dogpyr.resize(nOctaves);
    for (int i = 0; i < nOctaves; ++i)
        dogpyr[i].resize(nLayers);

#pragma omp parallel for num_threads(nthreads) schedule(dynamic) default(none) shared(nOctaves, nLayers, gpyr, dogpyr)
	for (int i = 0; i < nOctaves; i++)
		for (int j = 0; j < nLayers; j++)
             cv::subtract(gpyr[i][j + 1], gpyr[i][j], dogpyr[i][j], cv::noArray(), CV_32FC1);
}

float
SiftOmp::calc_orientation_hist(const cv::Mat &image, cv::Point pt, float scale, int n, float *hist) const {
    int radius = cvRound(SIFT_ORI_RADIUS * scale);  //特征点邻域半径(3*1.5*scale)
	int len = (2 * radius + 1) * (2 * radius + 1);       //特征点邻域像素总个数（最大值）
	float sigma = SIFT_ORI_RADIUS*scale;                 //特征点邻域高斯权重标准差(1.5*scale)
	float exp_scale = -1.f / (2 * sigma * sigma);
	cv::AutoBuffer<float> buffer(5 * len + n + 4);  //使用AutoBuffer分配一段内存，这里多出4个空间的目的是为了方便后面平滑直方图的需要
	float *X = buffer, *Y = X + len, *Mag = Y + len, *Ori = Mag + len, *W = Ori + len;  	//X保存水平差分，Y保存数值差分，Mag保存梯度幅度，Ori保存梯度角度，W保存高斯权重
	float *temp_hist = W + len + 2;                     //临时保存直方图数据
	for (int i = 0; i < n; i++)
		temp_hist[i] = 0.f;//数据清零

    //计算邻域像素的水平差分和竖直差分
    int k = 0;  // local counter for points within radius
    int x, y;
    float dx, dy;
    for (int i = -radius; i <= radius; ++i)
    {
        y = pt.y + i;
        if (y <= 0 || y >= image.rows - 1)
            continue;
        for (int j = -radius; j < radius; ++j) {
            x = pt.x + j;
            if (x <= 0 || x >= image.cols - 1)
                continue;
            dx = image.at<float>(y, x + 1) - image.at<float>(y, x - 1);
            dy = image.at<float>(y + 1, x) - image.at<float>(y - 1, x);
            X[k] = dx;
            Y[k] = dy;
            W[k] = (i * i + j * j) * exp_scale;
            ++k;
        }
    }

    //计算邻域像素的梯度幅度,梯度方向，高斯权重
    len = k;
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    // 计算直方图
    for (int i = 0; i < len; i++) {
        int bin = cvRound((n / 360.f) * Ori[i]);    //bin的范围约束在[0,(n-1)]
        if (bin >= n)
            bin -= n;
        if (bin < 0)
            bin += n;
        temp_hist[bin] += Mag[i] * W[i];
    }

    //平滑直方图
    temp_hist[-1] = temp_hist[n - 1];
    temp_hist[-2] = temp_hist[n - 2];
    temp_hist[n] = temp_hist[0];
    temp_hist[n + 1] = temp_hist[1];
    for (int i = 0; i < n; i++)
        hist[i] = (temp_hist[i - 2] + temp_hist[i + 2]) * 0.0625f +
                  (temp_hist[i - 1] + temp_hist[i + 1]) * 0.25f +
                  temp_hist[i] * 0.375f;
    float max_value = hist[0];
    for (int i = 1; i < n; ++i)
        if (hist[i] > max_value)
            max_value = hist[i];

    return max_value;
}

void SiftOmp::find_scale_space_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr,
                                       const std::vector<std::vector<cv::Mat>> &gpyr,
                                       std::vector<cv::KeyPoint> &kpts) const {
    kpts.clear();

    // find local extrema depending on the contrast
    std::vector<LocalExtrema> extrema;
    find_local_extrema(dogpyr, extrema);

    // adjust local extrema
    std::vector<LocalExtrema> extrema_adjust;
    std::vector<cv::KeyPoint> kpts_mid;
    adjust(dogpyr, extrema, kpts_mid, extrema_adjust);

    // calc_orientation
    calc_orientation(gpyr, kpts_mid, extrema_adjust, kpts);
}

void SiftOmp::detect(const cv::Mat &image, std::vector<std::vector<cv::Mat>> &gpyr,
                     std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint> &kpts) const {
    if (image.empty() || image.depth() != CV_8U)
		CV_Error(cv::Error::StsBadArg,"输入图像为空，或者图像深度不是CV_8U");

	//计算高斯金字塔组数
	int nOctaves = num_octaves(image);

	//生成高斯金字塔第一层图像
	cv::Mat init_gauss;
	create_init_img(image, init_gauss);

	//生成高斯尺度空间图像
	build_gaussian_pyramid(init_gauss, gpyr, nOctaves);

	//生成高斯差分金字塔(DOG金字塔，or LOG金字塔)
	build_dog_pyramid(dogpyr, gpyr);

	//在DOG金字塔上检测特征点
	find_scale_space_extrema(dogpyr, gpyr, kpts);

	if (nfeatures != 0 && nfeatures < (int)kpts.size())
	{
		sort(kpts.begin(), kpts.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b)
		{
            return a.response>b.response;
        });
		kpts.erase(kpts.begin() + nfeatures, kpts.end());
	}
    if (kpts.size() == 0)
        std::cout << "No keypoints are detected!" << std::endl;

    for (auto &kpt : kpts) {        // 重新调整特征点的坐标，若初始图像经过上采样，则特征点坐标是相对于上采样图像的，因此调整回相对于原图像
        kpt.pt /= 2.f;
    }
}

void SiftOmp::calc_sift_descriptor(const cv::Mat &gauss_image, float main_angle, cv::Point2f pt, float scale,
                                   int d, int n, float *desc) const {
    Sift::calc_sift_descriptor(gauss_image, main_angle, pt, scale, d, n, desc);
}

void SiftOmp::compute(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<cv::KeyPoint> &keypoints,
                      cv::Mat &descriptors) const {
    int d = SIFT_DESCR_WIDTH;//d=4,特征点邻域网格个数是d x d
	int n = SIFT_DESCR_HIST_BINS;//n=8,每个网格特征点梯度角度等分为8个方向
    int kpts_num = (int)keypoints.size();
	descriptors.create(kpts_num, d*d*n, CV_32FC1);//分配空间

    std::vector<cv::KeyPoint> kpts;
    if (firstOctave < 0) {  // keypoints中的特征点位置是相对于原图片而言的，因此这里需要将特征点的位置扩大2倍
        kpts.resize(kpts_num);
        for (size_t i = 0; i < kpts_num; ++i) {
            kpts[i].pt = keypoints[i].pt * 2.f;
            kpts[i].octave = keypoints[i].octave;
            kpts[i].size = keypoints[i].size * 2.f;
            kpts[i].angle = keypoints[i].angle;
        }
    } else {
        kpts = keypoints;   // shallow copy
    }
#pragma omp parallel for num_threads(nthreads) schedule(dynamic) default(none) shared(kpts_num, kpts, gpyr, descriptors, d, n)
	for (size_t i = 0; i < kpts_num; ++i)//对于每一个特征点
	{
		//得到特征点所在的组号，层号
		int octaves = kpts[i].octave & 255;
		int layer = (kpts[i].octave >> 8) & 255;

		//得到特征点相对于本组的坐标，不是最底层
		cv::Point2f pt(kpts[i].pt.x / (float)(1 << octaves), kpts[i].pt.y / (float)(1 << octaves));
		float scale = kpts[i].size / (float)(1 << octaves);//得到特征点相对于本组的尺度
		float main_angle = kpts[i].angle;//特征点主方向

		//计算该点的描述子
		calc_sift_descriptor(gpyr[octaves][layer], main_angle, pt, scale, d, n, descriptors.ptr<float>((int)i));
	}
}

void SiftOmp::find_local_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<LocalExtrema> &extrema) const {
    extrema.clear();
    float threshold = (float)(contrastThreshold / nOctaveLayers);
    int nOctaves = dogpyr.size();
    std::vector<std::vector<LocalExtrema>> threads_extrema(nthreads);
#pragma omp parallel default(none) shared(nOctaves, nOctaveLayers, dogpyr, threshold, threads_extrema)
    for (int i = 0; i < nOctaves; ++i)//对于每一组
    {
        for (int j = 1; j <= nOctaveLayers; ++j)//对于组内每一层
        {
            const cv::Mat &curr_img = dogpyr[i][j];//当前层
            const cv::Mat &prev_img = dogpyr[i][j - 1];//之前层
            const cv::Mat &next_img = dogpyr[i][j + 1];
            int num_row = curr_img.rows;
            int num_col = curr_img.cols;//获得当前组图像的大小
            size_t step = curr_img.step1();//一行元素所占宽度
#pragma omp for schedule(dynamic) nowait
            for (int r = SIFT_IMG_BORDER; r < num_row - SIFT_IMG_BORDER; ++r) {
                const float *curr_ptr = curr_img.ptr<float>(r);
                const float *prev_ptr = prev_img.ptr<float>(r);
                const float *next_ptr = next_img.ptr<float>(r);

                for (int c = SIFT_IMG_BORDER; c < num_col - SIFT_IMG_BORDER; ++c) {
                    float val = curr_ptr[c];
                    bool is_extrema = false;
                    is_extrema = (abs(val) > threshold &&
                                  ((val > 0 && val >= curr_ptr[c - 1] && val >= curr_ptr[c + 1] &&
                                    val >= curr_ptr[c - step - 1] && val >= curr_ptr[c - step] &&
                                    val >= curr_ptr[c - step + 1] &&
                                    val >= curr_ptr[c + step - 1] && val >= curr_ptr[c + step] &&
                                    val >= curr_ptr[c + step + 1] &&

                                    val >= prev_ptr[c] && val >= prev_ptr[c - 1] && val >= prev_ptr[c + 1] &&
                                    val >= prev_ptr[c - step - 1] && val >= prev_ptr[c - step] &&
                                    val >= prev_ptr[c - step + 1] &&
                                    val >= prev_ptr[c + step - 1] && val >= prev_ptr[c + step] &&
                                    val >= prev_ptr[c + step + 1] &&

                                    val >= next_ptr[c] && val >= next_ptr[c - 1] && val >= next_ptr[c + 1] &&
                                    val >= next_ptr[c - step - 1] && val >= next_ptr[c - step] &&
                                    val >= next_ptr[c - step + 1] &&
                                    val >= next_ptr[c + step - 1] && val >= next_ptr[c + step] &&
                                    val >= next_ptr[c + step + 1])
                                   ||
                                   (val < 0 && val <= curr_ptr[c - 1] && val <= curr_ptr[c + 1] &&
                                    val <= curr_ptr[c - step - 1] && val <= curr_ptr[c - step] &&
                                    val <= curr_ptr[c - step + 1] &&
                                    val <= curr_ptr[c + step - 1] && val <= curr_ptr[c + step] &&
                                    val <= curr_ptr[c + step + 1] &&

                                    val <= prev_ptr[c] && val <= prev_ptr[c - 1] && val <= prev_ptr[c + 1] &&
                                    val <= prev_ptr[c - step - 1] && val <= prev_ptr[c - step] &&
                                    val <= prev_ptr[c - step + 1] &&
                                    val <= prev_ptr[c + step - 1] && val <= prev_ptr[c + step] &&
                                    val <= prev_ptr[c + step + 1] &&

                                    val <= next_ptr[c] && val <= next_ptr[c - 1] && val <= next_ptr[c + 1] &&
                                    val <= next_ptr[c - step - 1] && val <= next_ptr[c - step] &&
                                    val <= next_ptr[c - step + 1] &&
                                    val <= next_ptr[c + step - 1] && val <= next_ptr[c + step] &&
                                    val <= next_ptr[c + step + 1])));
                    if (is_extrema) {
                        threads_extrema[omp_get_thread_num()].emplace_back(
                                LocalExtrema{
                                    .r = r,
                                    .c = c,
                                    .octave = i,
                                    .layer = j
                                });
                    }
                }
            }
        }
    }
    for (auto &e : threads_extrema) {
        extrema.insert(extrema.end(), e.begin(), e.end());
    }

    // DEBUG:
//    std::cout << "sift_mt detect " << extrema.size() << " local extrema" << std::endl;
}

void
SiftOmp::adjust(const std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<LocalExtrema> &extrema,
                std::vector<cv::KeyPoint> &kpts, std::vector<LocalExtrema> &extrema_adjust) const {
    kpts.clear();
    extrema_adjust.clear();
    int nExtrema = extrema.size();
    std::vector<std::vector<cv::KeyPoint>> threads_kpts(nthreads);
    std::vector<std::vector<LocalExtrema>> threads_extrema_adjust(nthreads);
#pragma omp parallel for schedule(dynamic) default(none) shared(nExtrema, extrema, dogpyr, threads_kpts, threads_extrema_adjust)
    for (int j = 0; j < nExtrema; j++) {
        int thread_id = omp_get_thread_num();
        const LocalExtrema &e = extrema[j];
        int octave = e.octave;
        int layer = e.layer;
        int row = e.r;
        int col = e.c;
        float xi = 0, xr = 0, xc = 0;
        int i = 0;
        bool not_take = false;
        for (; i < SIFT_MAX_INTERP_STEPS; ++i)//最大迭代次数
        {
            const cv::Mat &img = dogpyr[octave][layer];//当前层图像索引
            const cv::Mat &prev = dogpyr[octave][layer - 1];//之前层图像索引
            const cv::Mat &next = dogpyr[octave][layer + 1];//下一层图像索引

            //特征点位置x方向，y方向,尺度方向的一阶偏导数
            float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * 0.5f;
            float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * 0.5f;
            float dz = (next.at<float>(row, col) - prev.at<float>(row, col)) * 0.5f;

            //计算特征点位置二阶偏导数
            float v2 = img.at<float>(row, col);
            float dxx = img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
            float dyy = img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
            float dzz = prev.at<float>(row, col) + next.at<float>(row, col) - 2 * v2;

            //计算特征点周围混合二阶偏导数
            float dxy = (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
                         img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) * 0.25f;
            float dxz = (next.at<float>(row, col + 1) + prev.at<float>(row, col - 1) -
                         next.at<float>(row, col - 1) - prev.at<float>(row, col + 1)) * 0.25f;
            float dyz = (next.at<float>(row + 1, col) + prev.at<float>(row - 1, col) -
                         next.at<float>(row - 1, col) - prev.at<float>(row + 1, col)) * 0.25f;

            cv::Matx33f H(dxx, dxy, dxz,
                          dxy, dyy, dyz,
                          dxz, dyz, dzz);

            cv::Vec3f dD(dx, dy, dz);

            // 用Hessian矩阵的逆矩阵乘以梯度向量，得到偏移量
            cv::Vec3f X = H.solve(dD, cv::DECOMP_SVD);

            xc = -X[0];//x方向偏移量
            xr = -X[1];//y方向偏移量
            xi = -X[2];//尺度方向偏移量

            //如果三个方向偏移量都小于0.5，说明已经找到特征点准确位置
            if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xi) < 0.5f) {
                break;
            }

            //如果其中一个方向的偏移量过大，则删除该点
            if (abs(xc) > (float) (INT_MAX / 3) ||
                abs(xr) > (float) (INT_MAX / 3) ||
                abs(xi) > (float) (INT_MAX / 3)) {
                not_take = true;
                break;
            }

            col += cvRound(xc);
            row += cvRound(xr);
            layer += cvRound(xi);

            //如果特征点定位在边界区域，同样也需要删除
            if (layer < 1 || layer > nOctaveLayers ||
                col < SIFT_IMG_BORDER || col > img.cols - SIFT_IMG_BORDER ||
                row < SIFT_IMG_BORDER || row > img.rows - SIFT_IMG_BORDER) {
                not_take = true;
                break;
            }
        }
        if (not_take || i >= SIFT_MAX_INTERP_STEPS) {
            continue;   // adjust the next extrema
        }

        // 再次删除低响应点
        //再次计算特征点位置x方向，y方向,尺度方向的一阶偏导数
        const cv::Mat &img = dogpyr[octave][layer];
        const cv::Mat &prev = dogpyr[octave][layer - 1];
        const cv::Mat &next = dogpyr[octave][layer + 1];

        float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * 0.5f;
        float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * 0.5f;
        float dz = (next.at<float>(row, col) - prev.at<float>(row, col)) * 0.5f;
        cv::Matx31f dD(dx, dy, dz);
        float t = dD.dot(cv::Matx31f(xc, xr, xi));

        float contr = img.at<float>(row, col) + t * 0.5f;//特征点响应
        if (abs(contr) < contrastThreshold / (float) nOctaveLayers) //Low建议contr阈值是0.03，但是RobHess等建议阈值为0.04/nOctaveLayers
            continue;

        // 删除边缘响应比较强的点
        //再次计算特征点位置二阶偏导数
        float v2 = img.at<float>(row, col);
        float dxx = img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
        float dyy = img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
        float dxy = (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
                     img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) * 0.25f;
        float det = dxx * dyy - dxy * dxy;
        float trace = dxx + dyy;
        if (det < 0 || (trace * trace * edgeThreshold >= det * (edgeThreshold + 1) * (edgeThreshold + 1)))
            continue;

        // 保存该特征点信息
        threads_kpts[thread_id].emplace_back(
                ((float) col + xc) * (float) (1 << octave),
                ((float) row + xr) * (float) (1 << octave),
                sigma * powf(2.f, ((float) layer + xi) / (float) nOctaveLayers) * (float) (1 << octave),
                -1,
                abs(contr),
                octave + (layer << 8)
                );
        threads_extrema_adjust[thread_id].emplace_back(
                LocalExtrema{
                    .r = row,
                    .c = col,
                    .octave = octave,
                    .layer = layer
                });
    }
    for (int i = 0; i < nthreads; i++) {
        kpts.insert(kpts.end(), threads_kpts[i].begin(), threads_kpts[i].end());
        extrema_adjust.insert(extrema_adjust.end(), threads_extrema_adjust[i].begin(), threads_extrema_adjust[i].end());
    }
    // DEBUG
//    std::cout << "after adjust, remains " << kpts.size() << " kpts" << std::endl;
}

void
SiftOmp::calc_orientation(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<cv::KeyPoint> &kpts,
                          const std::vector<LocalExtrema> &extrema, std::vector<cv::KeyPoint> &kpts_res) const {
    int nKpts = kpts.size();
    int n = SIFT_ORI_HIST_BINS;
    std::vector<std::vector<cv::KeyPoint>> threads_kpts(nthreads);
#pragma omp parallel for num_threads(nthreads) schedule(dynamic) default(none) shared(nKpts, kpts, extrema, gpyr, threads_kpts, n, SIFT_ORI_PEAK_RATIO)
    for (int j = 0; j < nKpts; j++) {
        const LocalExtrema &e = extrema[j];
        cv::KeyPoint kpt = kpts[j];
        int r = e.r;
        int c = e.c;
        int octave = e.octave;
        int layer = e.layer;
        float hist[n];
        float scale = kpts[j].size / float(1 << octave);
        float max_hist = calc_orientation_hist(gpyr[octave][layer], cv::Point(c, r), scale, n, hist);
        float mag_thr = max_hist * SIFT_ORI_PEAK_RATIO;
        for (int i = 0; i < n; ++i) {
            int left = 0, right = 0;
            if (i == 0)
                left = n - 1;
            else
                left = i - 1;
            if (i == n - 1)
                right = 0;
            else
                right = i + 1;
            if (hist[i] > hist[left] && hist[i] > hist[right] && hist[i] >= mag_thr) {
                float bin = i + 0.5f * (hist[left] - hist[right]) / (hist[left] + hist[right] - 2 * hist[i]);
                if (bin < 0)
                    bin += n;
                if (bin >= n)
                    bin -= n;

                kpt.angle = (360.f / n) * bin;//特征点的主方向0-360度
                threads_kpts[omp_get_thread_num()].emplace_back(kpt.pt, kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id);
            }
        }
    }
    for (int i = 0; i < nthreads; i++) {
        kpts_res.insert(kpts_res.end(), threads_kpts[i].begin(), threads_kpts[i].end());
    }
}

void SiftOpencvPara::build_gaussian_pyramid(const cv::Mat &init_img, std::vector<std::vector<cv::Mat>> &gpyr,
                                     int nOctaves) const {
    const int nLayers = nOctaveLayers + 3;
    std::vector<double> sig(nLayers);
    gpyr.resize(nOctaves);
    for (int i = 0; i < nOctaves; i++)
        gpyr[i].resize(nLayers);

    sig[0] = sigma;
    double k = pow(2.0, 1.0 / nOctaveLayers);

    cv::parallel_for_(cv::Range(1, nLayers), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; i++) {
            double prev = pow(k, double(i - 1)) * sigma;
            double curr = k * prev;
            sig[i] = sqrt(curr * curr - prev * prev);
        }
    });

    // 该层循环存在数据依赖问题，并行需要进一步研究，可否采用直接求不相邻高斯图像的方法？
    // TODO
    for (int i = 0; i < nOctaves; i++) {
        for (int j = 0; j < nLayers; j++) {
            if (i == 0 && j == 0) {
                gpyr[0][0] = init_img;
            } else if (j == 0) {
                cv::Mat &prev = gpyr[i - 1][nOctaveLayers];
                cv::resize(prev, gpyr[i][0], cv::Size(prev.cols >> 1, prev.rows >> 1), 0, 0, cv::INTER_NEAREST);
            } else {
                int ksize = 2 * cvRound(SIFT_GAUSS_KERNEL_RATIO * sig[j]) + 1;
                cv::GaussianBlur(gpyr[i][j - 1], gpyr[i][j], cv::Size(ksize, ksize), sig[j], sig[j]);
            }
        }
    }
}

void SiftOpencvPara::build_dog_pyramid(std::vector<std::vector<cv::Mat>> &dogpyr,
                                const std::vector<std::vector<cv::Mat>> &gpyr) const {
    int nOctaves = gpyr.size();
    int nLayers = nOctaveLayers + 2;
    dogpyr.resize(nOctaves);
    for (int i = 0; i < nOctaves; ++i)
        dogpyr[i].resize(nLayers);

    cv::parallel_for_(cv::Range(0, nOctaves), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; i++) {
            for (int j = 0; j < nLayers; j++) {
                cv::subtract(gpyr[i][j + 1], gpyr[i][j], dogpyr[i][j], cv::noArray(), CV_32FC1);
            }
        }
    });
}

float
SiftOpencvPara::calc_orientation_hist(const cv::Mat &image, cv::Point pt, float scale, int n, float *hist) const {
    int radius = cvRound(SIFT_ORI_RADIUS * scale);  //特征点邻域半径(3*1.5*scale)
	int len = (2 * radius + 1) * (2 * radius + 1);       //特征点邻域像素总个数（最大值）
	float sigma = SIFT_ORI_RADIUS*scale;                 //特征点邻域高斯权重标准差(1.5*scale)
	float exp_scale = -1.f / (2 * sigma * sigma);
	cv::AutoBuffer<float> buffer(5 * len + n + 4);  //使用AutoBuffer分配一段内存，这里多出4个空间的目的是为了方便后面平滑直方图的需要
	float *X = buffer, *Y = X + len, *Mag = Y + len, *Ori = Mag + len, *W = Ori + len;  	//X保存水平差分，Y保存数值差分，Mag保存梯度幅度，Ori保存梯度角度，W保存高斯权重
	float *temp_hist = W + len + 2;                     //临时保存直方图数据
	for (int i = 0; i < n; i++)
		temp_hist[i] = 0.f;//数据清零

    //计算邻域像素的水平差分和竖直差分
    int k = 0;  // local counter for points within radius
    int x, y;
    float dx, dy;
    for (int i = -radius; i <= radius; ++i)
    {
        y = pt.y + i;
        if (y <= 0 || y >= image.rows - 1)
            continue;
        for (int j = -radius; j < radius; ++j) {
            x = pt.x + j;
            if (x <= 0 || x >= image.cols - 1)
                continue;
            dx = image.at<float>(y, x + 1) - image.at<float>(y, x - 1);
            dy = image.at<float>(y + 1, x) - image.at<float>(y - 1, x);
            X[k] = dx;
            Y[k] = dy;
            W[k] = (i * i + j * j) * exp_scale;
            ++k;
        }
    }

    //计算邻域像素的梯度幅度,梯度方向，高斯权重
    len = k;
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    // 计算直方图
    for (int i = 0; i < len; i++) {
        int bin = cvRound((n / 360.f) * Ori[i]);    //bin的范围约束在[0,(n-1)]
        if (bin >= n)
            bin -= n;
        if (bin < 0)
            bin += n;
        temp_hist[bin] += Mag[i] * W[i];
    }

    //平滑直方图
    temp_hist[-1] = temp_hist[n - 1];
    temp_hist[-2] = temp_hist[n - 2];
    temp_hist[n] = temp_hist[0];
    temp_hist[n + 1] = temp_hist[1];
    for (int i = 0; i < n; i++)
        hist[i] = (temp_hist[i - 2] + temp_hist[i + 2]) * 0.0625f +
                  (temp_hist[i - 1] + temp_hist[i + 1]) * 0.25f +
                  temp_hist[i] * 0.375f;
    float max_value = hist[0];
    for (int i = 1; i < n; ++i)
        if (hist[i] > max_value)
            max_value = hist[i];

    return max_value;
}

void SiftOpencvPara::find_scale_space_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr,
                                       const std::vector<std::vector<cv::Mat>> &gpyr,
                                       std::vector<cv::KeyPoint> &kpts) const {
    kpts.clear();

    // find local extrema depending on the contrast
    std::vector<LocalExtrema> extrema;
    find_local_extrema(dogpyr, extrema);

    // adjust local extrema
    std::vector<LocalExtrema> extrema_adjust;
    std::vector<cv::KeyPoint> kpts_mid;
    adjust(dogpyr, extrema, kpts_mid, extrema_adjust);

    // calc_orientation
    calc_orientation(gpyr, kpts_mid, extrema_adjust, kpts);
}

void SiftOpencvPara::detect(const cv::Mat &image, std::vector<std::vector<cv::Mat>> &gpyr,
                     std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint> &kpts) const {
    if (image.empty() || image.depth() != CV_8U)
		CV_Error(cv::Error::StsBadArg,"输入图像为空，或者图像深度不是CV_8U");

	//计算高斯金字塔组数
	int nOctaves = num_octaves(image);

	//生成高斯金字塔第一层图像
	cv::Mat init_gauss;
	create_init_img(image, init_gauss);

	//生成高斯尺度空间图像
	build_gaussian_pyramid(init_gauss, gpyr, nOctaves);

	//生成高斯差分金字塔(DOG金字塔，or LOG金字塔)
	build_dog_pyramid(dogpyr, gpyr);

	//在DOG金字塔上检测特征点
	find_scale_space_extrema(dogpyr, gpyr, kpts);

	if (nfeatures != 0 && nfeatures < (int)kpts.size())
	{
		sort(kpts.begin(), kpts.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b)
		{
            return a.response>b.response;
        });
		kpts.erase(kpts.begin() + nfeatures, kpts.end());
	}
    if (kpts.size() == 0)
        std::cout << "No keypoints are detected!" << std::endl;

    for (auto &kpt : kpts) {        // 重新调整特征点的坐标，若初始图像经过上采样，则特征点坐标是相对于上采样图像的，因此调整回相对于原图像
        kpt.pt /= 2.f;
    }
}

void SiftOpencvPara::calc_sift_descriptor(const cv::Mat &gauss_image, float main_angle, cv::Point2f pt, float scale,
                                   int d, int n, float *desc) const {
    Sift::calc_sift_descriptor(gauss_image, main_angle, pt, scale, d, n, desc);
}

void SiftOpencvPara::compute(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<cv::KeyPoint> &keypoints,
                      cv::Mat &descriptors) const {
    int d = SIFT_DESCR_WIDTH;//d=4,特征点邻域网格个数是d x d
	int n = SIFT_DESCR_HIST_BINS;//n=8,每个网格特征点梯度角度等分为8个方向
    int kpts_num = (int)keypoints.size();
	descriptors.create(kpts_num, d*d*n, CV_32FC1);//分配空间

    std::vector<cv::KeyPoint> kpts;
    if (firstOctave < 0) {  // keypoints中的特征点位置是相对于原图片而言的，因此这里需要将特征点的位置扩大2倍
        kpts.resize(kpts_num);
        for (size_t i = 0; i < kpts_num; ++i) {
            kpts[i].pt = keypoints[i].pt * 2.f;
            kpts[i].octave = keypoints[i].octave;
            kpts[i].size = keypoints[i].size * 2.f;
            kpts[i].angle = keypoints[i].angle;
        }
    } else {
        kpts = keypoints;   // shallow copy
    }
    cv::parallel_for_(cv::Range(0, kpts_num), [&](const cv::Range &range) {
       for (size_t i = range.start; i < range.end; ++i)//对于每一个特征点
        {
            //得到特征点所在的组号，层号
            int octaves = kpts[i].octave & 255;
            int layer = (kpts[i].octave >> 8) & 255;

            //得到特征点相对于本组的坐标，不是最底层
            cv::Point2f pt(kpts[i].pt.x / (float)(1 << octaves), kpts[i].pt.y / (float)(1 << octaves));
            float scale = kpts[i].size / (float)(1 << octaves);//得到特征点相对于本组的尺度
            float main_angle = kpts[i].angle;//特征点主方向

            //计算该点的描述子
            calc_sift_descriptor(gpyr[octaves][layer], main_angle, pt, scale, d, n, descriptors.ptr<float>((int)i));
        }
    });

}

void SiftOpencvPara::find_local_extrema(const std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<LocalExtrema> &extrema) const {
    extrema.clear();
    float threshold = (float)(contrastThreshold / nOctaveLayers);
    int nOctaves = dogpyr.size();
    MyList<LocalExtrema> extrema_list[nthreads];
    for (int i = 0; i < nOctaves; ++i)//对于每一组
    {
        for (int j = 1; j <= nOctaveLayers; ++j)//对于组内每一层
        {
            const cv::Mat &curr_img = dogpyr[i][j];
            const cv::Mat &prev_img = dogpyr[i][j - 1];
            const cv::Mat &next_img = dogpyr[i][j + 1];
            int num_row = curr_img.rows;
            int num_col = curr_img.cols;//获得当前组图像的大小
            size_t step = curr_img.step1();//一行元素所占宽度
            cv::parallel_for_(cv::Range(SIFT_IMG_BORDER, num_row - SIFT_IMG_BORDER), [&](const cv::Range &range) {
                for (int r = range.start; r < range.end; r++) {
                    const float *curr_ptr = curr_img.ptr<float>(r);
                    const float *prev_ptr = prev_img.ptr<float>(r);
                    const float *next_ptr = next_img.ptr<float>(r);
                    for (int c = SIFT_IMG_BORDER; c < num_col - SIFT_IMG_BORDER; ++c) {
                        float val = curr_ptr[c];
                        bool is_extrema = false;
                        is_extrema = (abs(val) > threshold &&
                                      ((val > 0 && val >= curr_ptr[c - 1] && val >= curr_ptr[c + 1] &&
                                        val >= curr_ptr[c - step - 1] && val >= curr_ptr[c - step] &&
                                        val >= curr_ptr[c - step + 1] &&
                                        val >= curr_ptr[c + step - 1] && val >= curr_ptr[c + step] &&
                                        val >= curr_ptr[c + step + 1] &&

                                        val >= prev_ptr[c] && val >= prev_ptr[c - 1] && val >= prev_ptr[c + 1] &&
                                        val >= prev_ptr[c - step - 1] && val >= prev_ptr[c - step] &&
                                        val >= prev_ptr[c - step + 1] &&
                                        val >= prev_ptr[c + step - 1] && val >= prev_ptr[c + step] &&
                                        val >= prev_ptr[c + step + 1] &&

                                        val >= next_ptr[c] && val >= next_ptr[c - 1] && val >= next_ptr[c + 1] &&
                                        val >= next_ptr[c - step - 1] && val >= next_ptr[c - step] &&
                                        val >= next_ptr[c - step + 1] &&
                                        val >= next_ptr[c + step - 1] && val >= next_ptr[c + step] &&
                                        val >= next_ptr[c + step + 1])
                                       ||
                                       (val < 0 && val <= curr_ptr[c - 1] && val <= curr_ptr[c + 1] &&
                                        val <= curr_ptr[c - step - 1] && val <= curr_ptr[c - step] &&
                                        val <= curr_ptr[c - step + 1] &&
                                        val <= curr_ptr[c + step - 1] && val <= curr_ptr[c + step] &&
                                        val <= curr_ptr[c + step + 1] &&

                                        val <= prev_ptr[c] && val <= prev_ptr[c - 1] && val <= prev_ptr[c + 1] &&
                                        val <= prev_ptr[c - step - 1] && val <= prev_ptr[c - step] &&
                                        val <= prev_ptr[c - step + 1] &&
                                        val <= prev_ptr[c + step - 1] && val <= prev_ptr[c + step] &&
                                        val <= prev_ptr[c + step + 1] &&

                                        val <= next_ptr[c] && val <= next_ptr[c - 1] && val <= next_ptr[c + 1] &&
                                        val <= next_ptr[c - step - 1] && val <= next_ptr[c - step] &&
                                        val <= next_ptr[c - step + 1] &&
                                        val <= next_ptr[c + step - 1] && val <= next_ptr[c + step] &&
                                        val <= next_ptr[c + step + 1])));
                        if (is_extrema) {
                            LocalExtrema e = {
                                    .r = r,
                                    .c = c,
                                    .octave = i,
                                    .layer = j
                            };
                            int thread_id = cv::getThreadNum();
                            std::cout << thread_id << std::endl;
//                            extrema_list[thread_id].push_back(e);
                        }
                    }
                }
            });
        }
    }
    for (int i = 0; i < nthreads; i++) {
        std::vector<LocalExtrema> &thread_extrema = extrema_list[i].data;
        extrema.insert(extrema.end(), thread_extrema.begin(), thread_extrema.end());
    }

    // DEBUG:
//    std::cout << "sift_mt detect " << extrema.size() << " local extrema" << std::endl;
}

void
SiftOpencvPara::adjust(const std::vector<std::vector<cv::Mat>> &dogpyr, const std::vector<LocalExtrema> &extrema,
                std::vector<cv::KeyPoint> &kpts, std::vector<LocalExtrema> &extrema_adjust) const {
    kpts.clear();
    extrema_adjust.clear();
    int nExtrema = extrema.size();
    std::vector<std::vector<cv::KeyPoint>> threads_kpts(nthreads);
    std::vector<std::vector<LocalExtrema>> threads_extrema_adjust(nthreads);
    cv::parallel_for_(cv::Range(0, nExtrema), [&](const cv::Range &range) {
        for (int j = range.start; j < range.end; j++) {
            int thread_id = cv::getThreadNum();
            const LocalExtrema &e = extrema[j];
            int octave = e.octave;
            int layer = e.layer;
            int row = e.r;
            int col = e.c;
            float xi = 0, xr = 0, xc = 0;
            int i = 0;
            bool not_take = false;
            for (; i < SIFT_MAX_INTERP_STEPS; ++i)//最大迭代次数
            {
                const cv::Mat &img = dogpyr[octave][layer];//当前层图像索引
                const cv::Mat &prev = dogpyr[octave][layer - 1];//之前层图像索引
                const cv::Mat &next = dogpyr[octave][layer + 1];//下一层图像索引

                //特征点位置x方向，y方向,尺度方向的一阶偏导数
                float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * 0.5f;
                float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * 0.5f;
                float dz = (next.at<float>(row, col) - prev.at<float>(row, col)) * 0.5f;

                //计算特征点位置二阶偏导数
                float v2 = img.at<float>(row, col);
                float dxx = img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
                float dyy = img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
                float dzz = prev.at<float>(row, col) + next.at<float>(row, col) - 2 * v2;

                //计算特征点周围混合二阶偏导数
                float dxy = (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
                             img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) * 0.25f;
                float dxz = (next.at<float>(row, col + 1) + prev.at<float>(row, col - 1) -
                             next.at<float>(row, col - 1) - prev.at<float>(row, col + 1)) * 0.25f;
                float dyz = (next.at<float>(row + 1, col) + prev.at<float>(row - 1, col) -
                             next.at<float>(row - 1, col) - prev.at<float>(row + 1, col)) * 0.25f;

                cv::Matx33f H(dxx, dxy, dxz,
                              dxy, dyy, dyz,
                              dxz, dyz, dzz);

                cv::Vec3f dD(dx, dy, dz);

                // 用Hessian矩阵的逆矩阵乘以梯度向量，得到偏移量
                cv::Vec3f X = H.solve(dD, cv::DECOMP_SVD);

                xc = -X[0];//x方向偏移量
                xr = -X[1];//y方向偏移量
                xi = -X[2];//尺度方向偏移量

                //如果三个方向偏移量都小于0.5，说明已经找到特征点准确位置
                if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xi) < 0.5f) {
                    break;
                }

                //如果其中一个方向的偏移量过大，则删除该点
                if (abs(xc) > (float) (INT_MAX / 3) ||
                    abs(xr) > (float) (INT_MAX / 3) ||
                    abs(xi) > (float) (INT_MAX / 3)) {
                    not_take = true;
                    break;
                }

                col += cvRound(xc);
                row += cvRound(xr);
                layer += cvRound(xi);

                //如果特征点定位在边界区域，同样也需要删除
                if (layer < 1 || layer > nOctaveLayers ||
                    col < SIFT_IMG_BORDER || col > img.cols - SIFT_IMG_BORDER ||
                    row < SIFT_IMG_BORDER || row > img.rows - SIFT_IMG_BORDER) {
                    not_take = true;
                    break;
                }
            }
            if (not_take || i >= SIFT_MAX_INTERP_STEPS) {
                continue;   // adjust the next extrema
            }

            // 再次删除低响应点
            //再次计算特征点位置x方向，y方向,尺度方向的一阶偏导数
            const cv::Mat &img = dogpyr[octave][layer];
            const cv::Mat &prev = dogpyr[octave][layer - 1];
            const cv::Mat &next = dogpyr[octave][layer + 1];

            float dx = (img.at<float>(row, col + 1) - img.at<float>(row, col - 1)) * 0.5f;
            float dy = (img.at<float>(row + 1, col) - img.at<float>(row - 1, col)) * 0.5f;
            float dz = (next.at<float>(row, col) - prev.at<float>(row, col)) * 0.5f;
            cv::Matx31f dD(dx, dy, dz);
            float t = dD.dot(cv::Matx31f(xc, xr, xi));

            float contr = img.at<float>(row, col) + t * 0.5f;//特征点响应
            if (abs(contr) < contrastThreshold / (float) nOctaveLayers) //Low建议contr阈值是0.03，但是RobHess等建议阈值为0.04/nOctaveLayers
                continue;

            // 删除边缘响应比较强的点
            //再次计算特征点位置二阶偏导数
            float v2 = img.at<float>(row, col);
            float dxx = img.at<float>(row, col + 1) + img.at<float>(row, col - 1) - 2 * v2;
            float dyy = img.at<float>(row + 1, col) + img.at<float>(row - 1, col) - 2 * v2;
            float dxy = (img.at<float>(row + 1, col + 1) + img.at<float>(row - 1, col - 1) -
                         img.at<float>(row + 1, col - 1) - img.at<float>(row - 1, col + 1)) * 0.25f;
            float det = dxx * dyy - dxy * dxy;
            float trace = dxx + dyy;
            if (det < 0 || (trace * trace * edgeThreshold >= det * (edgeThreshold + 1) * (edgeThreshold + 1)))
                continue;

            // 保存该特征点信息
            threads_kpts[thread_id].emplace_back(
                    ((float) col + xc) * (float) (1 << octave),
                    ((float) row + xr) * (float) (1 << octave),
                    sigma * powf(2.f, ((float) layer + xi) / (float) nOctaveLayers) * (float) (1 << octave),
                    -1,
                    abs(contr),
                    octave + (layer << 8)
                    );
            threads_extrema_adjust[thread_id].emplace_back(
                    LocalExtrema{
                        .r = row,
                        .c = col,
                        .octave = octave,
                        .layer = layer
                    });
        }
    });

    for (int i = 0; i < nthreads; i++) {
        kpts.insert(kpts.end(), threads_kpts[i].begin(), threads_kpts[i].end());
        extrema_adjust.insert(extrema_adjust.end(), threads_extrema_adjust[i].begin(), threads_extrema_adjust[i].end());
    }
    // DEBUG
//    std::cout << "after adjust, remains " << kpts.size() << " kpts" << std::endl;
}

void
SiftOpencvPara::calc_orientation(const std::vector<std::vector<cv::Mat>> &gpyr, const std::vector<cv::KeyPoint> &kpts,
                          const std::vector<LocalExtrema> &extrema, std::vector<cv::KeyPoint> &kpts_res) const {
    int nKpts = kpts.size();
    int n = SIFT_ORI_HIST_BINS;
    std::vector<std::vector<cv::KeyPoint>> threads_kpts(nthreads);
    cv::parallel_for_(cv::Range(0, nKpts), [&](const cv::Range &range) {
        for (int j = range.start; j < range.end; j++) {
            const LocalExtrema &e = extrema[j];
            cv::KeyPoint kpt = kpts[j];
            int r = e.r;
            int c = e.c;
            int octave = e.octave;
            int layer = e.layer;
            float hist[n];
            float scale = kpts[j].size / float(1 << octave);
            float max_hist = calc_orientation_hist(gpyr[octave][layer], cv::Point(c, r), scale, n, hist);
            float mag_thr = max_hist * SIFT_ORI_PEAK_RATIO;
            for (int i = 0; i < n; ++i) {
                int left = 0, right = 0;
                if (i == 0)
                    left = n - 1;
                else
                    left = i - 1;
                if (i == n - 1)
                    right = 0;
                else
                    right = i + 1;
                if (hist[i] > hist[left] && hist[i] > hist[right] && hist[i] >= mag_thr) {
                    float bin = i + 0.5f * (hist[left] - hist[right]) / (hist[left] + hist[right] - 2 * hist[i]);
                    if (bin < 0)
                        bin += n;
                    if (bin >= n)
                        bin -= n;

                    kpt.angle = (360.f / n) * bin;//特征点的主方向0-360度
                    threads_kpts[cv::getThreadNum()].emplace_back(kpt.pt, kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id);
                }
            }
        }
    });
    for (int i = 0; i < nthreads; i++) {
        kpts_res.insert(kpts_res.end(), threads_kpts[i].begin(), threads_kpts[i].end());
    }
}
