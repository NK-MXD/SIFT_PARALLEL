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
void test_with_opencv_sift();

int main(int argc,char *argv[])
{
//	test_fusion();
    test_with_opencv_sift();
}

void test_fusion() {
    Mat image_1 = imread("/home/guo/mypro/CV/lab4-feature-extraction/ref/migrate-sift-opencv249/test_images/ucsb1.jpg", ImreadModes::IMREAD_UNCHANGED);
	Mat image_2 = imread("/home/guo/mypro/CV/lab4-feature-extraction/ref/migrate-sift-opencv249/test_images/ucsb2.jpg", ImreadModes::IMREAD_UNCHANGED);
//	Mat image_1 = imread("/home/guo/mypro/CV/lab4-feature-extraction/feature/assets/Lenna.png", ImreadModes::IMREAD_UNCHANGED);
//	Mat image_2 = imread("/home/guo/mypro/CV/lab4-feature-extraction/feature/assets/Lenna.png", ImreadModes::IMREAD_UNCHANGED);
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

    t_cal_m = (double)cv::getTickCount();
    sift_m->comput_des(gpyr, kpts_m, desc_m);
    t_cal_m = ((double)cv::getTickCount() - t_cal_m) / getTickFrequency();

    t_detect_o = (double)cv::getTickCount();
    sift_o->detectAndCompute(img_gray, cv::noArray(), kpts_o, desc_o);
    t_detect_o = ((double)cv::getTickCount() - t_detect_o) / getTickFrequency();

    t_cal_o = (double)cv::getTickCount();
    sift_o->compute(img_gray, kpts_o, desc_o);
    t_cal_o = ((double)cv::getTickCount() - t_cal_o) / getTickFrequency();

    std::cout << "MySift detect time: " << t_detect_m << " kpts size: " << kpts_m.size() << std::endl;
    std::cout << "MySift compute time: " << t_cal_m << " desc size: " << desc_m.rows << std::endl;
    std::cout << "OpenCV SIFT detect time: " << t_detect_o << " kpts size: " << kpts_o.size() << std::endl;
    std::cout << "OpenCV SIFT compute time: " << t_cal_o << " desc size: " << desc_o.rows << std::endl;
}