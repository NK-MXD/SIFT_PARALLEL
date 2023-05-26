#include"sift.h"
#include"display.h"
#include"match.h"

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<fstream>
#include<stdlib.h>
#include <filesystem>

void test_fusion();
void test_compute_and_detect();
void test_with_opencv_sift();
void test_compute_desc();
void test_match();

int main(int argc,char *argv[])
{
//	test_fusion();
//    test_compute_and_detect();
//    test_with_opencv_sift();
    test_compute_desc();
//    test_match();
}

void test_fusion() {
//    Mat image_1 = imread("/home/guo/mypro/CV/lab4-feature-extraction/ref/migrate-sift-opencv249/test_images/ucsb1.jpg", ImreadModes::IMREAD_UNCHANGED);
//	Mat image_2 = imread("/home/guo/mypro/CV/lab4-feature-extraction/ref/migrate-sift-opencv249/test_images/ucsb2.jpg", ImreadModes::IMREAD_UNCHANGED);
//	Mat image_1 = imread("/home/guo/mypro/CV/lab4-feature-extraction/feature/assets/Lenna.png", ImreadModes::IMREAD_UNCHANGED);
//	Mat image_2 = imread("/home/guo/mypro/CV/lab4-feature-extraction/feature/assets/Lenna.png", ImreadModes::IMREAD_UNCHANGED);
    Mat image_1 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/migrate-sift-opencv249/test_images/peacock.jpg");
    Mat image_2 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/migrate-sift-opencv249/test_images/peacock.jpg");
	string change_model = "perspective";

	//创建文件夹保存图像
	string newfile = "/home/guo/mypro/CV/lab4-feature-extraction/ref/migrate-sift-opencv249/image_save";
	// _mkdir(newfile);
	filesystem::create_directory(newfile);

	//算法运行总时间开始计时
	double total_count_beg = (double)getTickCount();

	//类对象
	MySift sift_1(0, 3, 0.04, 10, 1.6, true);

	//参考图像特征点检测和描述
	vector<vector<Mat>> gauss_pyr_1, dog_pyr_1;
	vector<KeyPoint> keypoints_1;
	Mat descriptors_1;
	double detect_1 = (double)getTickCount();
	sift_1.detect(image_1, gauss_pyr_1, dog_pyr_1, keypoints_1);
	double detect_time_1 = ((double)getTickCount() - detect_1) / getTickFrequency();
	cout << "参考图像特征点检测时间是： " << detect_time_1 << "s" << endl;
	cout << "参考图像检测特征点个数是： " << keypoints_1.size() << endl;

	double comput_1 = (double)getTickCount();
	sift_1.comput_des(gauss_pyr_1, keypoints_1, descriptors_1);
	double comput_time_1 = ((double)getTickCount() - comput_1) / getTickFrequency();
	cout << "参考图像特征点描述时间是： " << comput_time_1 << "s" << endl;


	//待配准图像特征点检测和描述
	vector<vector<Mat>> gauss_pyr_2, dog_pyr_2;
	vector<KeyPoint> keypoints_2;
	Mat descriptors_2;
	double detect_2 = (double)getTickCount();
	sift_1.detect(image_2, gauss_pyr_2, dog_pyr_2, keypoints_2);
	double detect_time_2 = ((double)getTickCount() - detect_2) / getTickFrequency();
	cout << "待配准图像特征点检测时间是： " << detect_time_2 << "s" << endl;
	cout << "待配准图像检测特征点个数是： " << keypoints_2.size() << endl;

    // draw keypoints
    Mat image_keypoints_1, image_keypoints_2;
    drawKeypoints(image_1, keypoints_1, image_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(image_2, keypoints_2, image_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    cv::imshow("Keypoints 1", image_keypoints_1);
    cv::imshow("Keypoints 2", image_keypoints_2);
    cv::waitKey(0);

	double comput_2 = (double)getTickCount();
	sift_1.comput_des(gauss_pyr_2, keypoints_2, descriptors_2);
	double comput_time_2 = ((double)getTickCount() - comput_2) / getTickFrequency();
	cout << "待配准特征点描述时间是： " << comput_time_2 << "s" << endl;

	//最近邻与次近邻距离比匹配
	double match_time = (double)getTickCount();
	Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher;
	//Ptr<DescriptorMatcher> matcher = new BFMatcher(NORM_L2);
	std::vector<vector<DMatch>> dmatchs;
	matcher->knnMatch(descriptors_1, descriptors_2, dmatchs, 2);
	//match_des(descriptors_1, descriptors_2, dmatchs, COS);

	Mat matched_lines;
	vector<DMatch> right_matchs;
	Mat homography = match(image_1, image_2, dmatchs, keypoints_1, keypoints_2, change_model,
		right_matchs,matched_lines);
	double match_time_2 = ((double)getTickCount() - match_time) / getTickFrequency();
	cout << "特征点匹配花费时间是： " << match_time_2 << "s" << endl;
	cout << change_model << "变换矩阵是：" << endl; 
	cout << homography << endl;

	//把正确匹配点坐标写入文件中
	ofstream ofile;
	ofile.open("/home/guo/mypro/CV/lab4-feature-extraction/ref/migrate-sift-opencv249/position.txt");
	for (size_t i = 0; i < right_matchs.size(); ++i)
	{
		ofile << keypoints_1[right_matchs[i].queryIdx].pt << "   "
			<< keypoints_2[right_matchs[i].trainIdx].pt << endl;
	}

	//图像融合
	double fusion_beg = (double)getTickCount();
	Mat fusion_image, mosaic_image, regist_image;
	image_fusion(image_1, image_2, homography, fusion_image, mosaic_image, regist_image);
	imwrite("/home/guo/mypro/CV/lab4-feature-extraction/ref/migrate-sift-opencv249/image_save/融合后的图像.jpg", fusion_image);
	imwrite("/home/guo/mypro/CV/lab4-feature-extraction/ref/migrate-sift-opencv249/image_save/融合后的镶嵌图像.jpg", mosaic_image);
	imwrite("/home/guo/mypro/CV/lab4-feature-extraction/ref/migrate-sift-opencv249/image_save/配准后的待配准图像.jpg", regist_image);
	double fusion_time = ((double)getTickCount() - fusion_beg) / getTickFrequency();
	cout << "图像融合花费时间是： " << fusion_time << "s" << endl;

	double total_time = ((double)getTickCount() - total_count_beg) / getTickFrequency();
	cout << "总花费时间是： " << total_time << "s" << endl;

	//显示匹配结果
	namedWindow("融合后的图像", WINDOW_AUTOSIZE);
	imshow("融合后的图像", fusion_image);
	namedWindow("融合镶嵌图像", WINDOW_AUTOSIZE);
	imshow("融合镶嵌图像", mosaic_image);
	stringstream s_2;
	string numstring_2, windowName;
	s_2 << right_matchs.size();
	s_2 >> numstring_2;
	windowName = string("正确点匹配连线图: ") + numstring_2;
	namedWindow(windowName, WINDOW_AUTOSIZE);
	imshow(windowName, matched_lines);

	//保存金字塔拼接好的金字塔图像
	//int nOctaveLayers = sift_1.get_nOctave_layers();
	//write_mosaic_pyramid(gauss_pyr_1, dog_pyr_1, gauss_pyr_2, dog_pyr_2, nOctaveLayers);

	waitKey(0);
}

void test_compute_and_detect() {
    MySift *sift_m;
    cv::Mat img, img_gray;
    std::vector<std::vector<cv::Mat>> gpyr, dogpyr;
    std::vector<cv::KeyPoint> kpts_m;
    cv::Mat desc_m;
    double t_detect_m, t_cal_m;

    img = cv::imread("/home/guo/mypro/SIFT_PARALLEL/migrate-sift-opencv249/test_images/peacock.jpg");
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    sift_m = new MySift();

    t_detect_m = (double)cv::getTickCount();
    sift_m->detect(img_gray, gpyr, dogpyr, kpts_m);
    t_detect_m = ((double)cv::getTickCount() - t_detect_m) / getTickFrequency();
    std::cout << "MySift detect time: " << t_detect_m << " kpts size: " << kpts_m.size() << std::endl;

    t_cal_m = (double)cv::getTickCount();
    sift_m->comput_des(gpyr, kpts_m, desc_m);
    t_cal_m = ((double)cv::getTickCount() - t_cal_m) / getTickFrequency();
    std::cout << "MySift compute time: " << t_cal_m << " desc size: " << desc_m.rows << std::endl;
}

void test_with_opencv_sift() {
    MySift *sift_m;
    cv::Ptr<cv::SIFT> sift_o;
    cv::Mat img, img_gray;
    std::vector<std::vector<cv::Mat>> gpyr, dogpyr;
    std::vector<cv::KeyPoint> kpts_m, kpts_o;
    cv::Mat desc_m, desc_o;
    double t_detect_m, t_cal_m, t_detect_o, t_cal_o;

    img = cv::imread("/home/guo/mypro/SIFT_PARALLEL/migrate-sift-opencv249/test_images/peacock.jpg");
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    sift_m = new MySift();
    sift_o = cv::SIFT::create();

    t_detect_m = (double)cv::getTickCount();
    sift_m->detect(img_gray, gpyr, dogpyr, kpts_m);
    t_detect_m = ((double)cv::getTickCount() - t_detect_m) / getTickFrequency();
    std::cout << "MySift detect time: " << t_detect_m << " kpts size: " << kpts_m.size() << std::endl;

    t_cal_m = (double)cv::getTickCount();
    sift_m->comput_des(gpyr, kpts_m, desc_m);
    t_cal_m = ((double)cv::getTickCount() - t_cal_m) / getTickFrequency();
    std::cout << "MySift compute time: " << t_cal_m << " desc size: " << desc_m.rows << std::endl;

    t_detect_o = (double)cv::getTickCount();
    sift_o->detect(img_gray, kpts_o);
    t_detect_o = ((double)cv::getTickCount() - t_detect_o) / getTickFrequency();
    std::cout << "OpenCV SIFT detect time: " << t_detect_o << " kpts size: " << kpts_o.size() << std::endl;

    t_cal_o = (double)cv::getTickCount();
    sift_o->compute(img_gray, kpts_o, desc_o);
    t_cal_o = ((double)cv::getTickCount() - t_cal_o) / getTickFrequency();
    std::cout << "OpenCV SIFT compute time: " << t_cal_o << " desc size: " << desc_o.rows << std::endl;

}

/* 测试的结果：并行加速的特征求解只比串行求解快一倍 */
void test_compute_desc() {
    MySift *sift;
    cv::Mat img, img_gray;
    std::vector<std::vector<cv::Mat>> gpyr, dogpyr;
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc_origin, desc_opencv_para;
    double t_detect, t_cal_origin, t_cal_opencv_para;

    img = cv::imread("/home/guo/mypro/SIFT_PARALLEL/migrate-sift-opencv249/test_images/peacock.jpg");
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    sift = new MySift();

    t_detect = (double)cv::getTickCount();
    sift->detect(img_gray, gpyr, dogpyr, kpts);
    t_detect = ((double)cv::getTickCount() - t_detect) / getTickFrequency();
    std::cout << "MySift detect time: " << t_detect << " kpts size: " << kpts.size() << std::endl;

    t_cal_opencv_para = (double)cv::getTickCount();
    sift->calc_descriptors_opencv_parallel_for(gpyr, kpts, desc_opencv_para);
    t_cal_opencv_para = ((double)cv::getTickCount() - t_cal_opencv_para) / getTickFrequency();
    std::cout << "MySift opencv `parallel_for` compute time: " << t_cal_opencv_para << std::endl;

    t_cal_origin = (double)cv::getTickCount();
    sift->calc_descriptors(gpyr, kpts, desc_origin);
    t_cal_origin = ((double)cv::getTickCount() - t_cal_origin) / getTickFrequency();
    std::cout << "MySift serial compute time: " << t_cal_origin << std::endl;
}

/* 测试的结果：在求特征子处进行粗粒度的并行化匹配效果与原算法相同 */
void test_match() {
    MySift *sift;
    cv::Mat img1, img2, img1_gray, img2_gray, img_match;
    std::vector<std::vector<cv::Mat>> gpyr1, dogpyr1, gpyr2, dogpyr2;
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good_matches;
    double t_detect, t_cal, t_match;

    sift = new MySift();
    matcher = cv::DescriptorMatcher::create("FlannBased");
    img1 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/migrate-sift-opencv249/test_images/test1.jpg");
    img2 = cv::imread("/home/guo/mypro/SIFT_PARALLEL/migrate-sift-opencv249/test_images/test2.jpg");
    cv::resize(img1, img1, cv::Size(), 0.25, 0.25);
    cv::resize(img2, img2, cv::Size(), 0.25, 0.25);
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    sift->detect(img1_gray, gpyr1, dogpyr1, kpts1);
    sift->detect(img2_gray, gpyr2, dogpyr2, kpts2);
    sift->calc_descriptors(gpyr1, kpts1, desc1);
    sift->calc_descriptors(gpyr2, kpts2, desc2);
//    sift->calc_descriptors_opencv_parallel_for(gpyr1, kpts1, desc1);
//    sift->calc_descriptors_opencv_parallel_for(gpyr2, kpts2, desc2);

    matcher->knnMatch(desc1, desc2, matches, 2);
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < 0.6 * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
    cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, img_match);
    cv::imshow("match", img_match);
    cv::waitKey(0);
}