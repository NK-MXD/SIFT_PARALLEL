#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <omp.h>
#include<sstream>
#include<vector>
#include<fstream>
#include<string.h>

#include "../Sar_sift.h"
#include "../match.h"

void test_sar_sift(){
    /*------------------------------------------test1: 特征点提取例子------------------------------------------*/
    /*
    Sar_sift *sar_sift;
    double t1, t2;
    double ratio = 0.6;

    cv::Mat img0, img0_gray;
    vector<Mat> sar_harris_fun0, amplit0, orient0;
    std::vector<cv::KeyPoint> kpts0;
    cv::Mat desc0;
    img0 = cv::imread("/work/home/acmhsiv3ds/SIFT_PARALLEL-sift-parallel/RSIR-SIFT/SAR-SIFT-opencv249/assets/peacock.jpg");
    cv::cvtColor(img0, img0_gray, cv::COLOR_BGR2GRAY);
    sar_sift = new Sar_sift(0, 8, 2, pow(2, 1.0 / 3.0), 0.8/5, 0.04);
    t1 = (double)cv::getTickCount();
    sar_sift->detect_keys(img0, kpts0, sar_harris_fun0, amplit0, orient0);
    t1 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
    std::cout << "sar_sift detect img0 time: " << t1 << "s, kpts0 size: " << kpts0.size() << std::endl;

    t2 = (double)cv::getTickCount();
    sar_sift->comput_des(kpts0, amplit0, orient0, desc0);
    t2 = ((double)cv::getTickCount() - t2) / cv::getTickFrequency();
    std::cout << "sar_sift compute img0 time: " << t2 << "s" << std::endl;
    std::cout << "sar_sift sift total time: " << t1 + t2 << "s" << std::endl;
    Mat sar_sift_keypoints;
    drawKeypoints(img0, kpts0, sar_sift_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // 保存图像
    imwrite("../output/sar_sift_keypoints.jpg", sar_sift_keypoints);

    // 标准测试
    // 标准sift生成的图像:
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

    // 检测关键点和计算描述符
    std::vector<cv::KeyPoint> keypoints00;
    cv::Mat descriptors00;
    double t0 = (double)cv::getTickCount();
    detector->detect(img0, keypoints00, cv::noArray());
    t0 = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    std::cout << "Standard sift detect time: " << t0 << "s, kpts0 size: " << keypoints00.size() << std::endl;
    double t01 = (double)cv::getTickCount();
    detector->compute(img0, keypoints00, descriptors00);
    // detector->detectAndCompute(img0, cv::noArray(), keypoints00, descriptors00);
    t01 = ((double)cv::getTickCount() - t01) / cv::getTickFrequency();
    std::cout << "Standard sift compute time: " << t01 << "s" << std::endl;
    std::cout << "Standard sift total time: " << (t0 + t01) << "s" << std::endl;
    std::cout << "Standard sift / Our SIFT: " << std::fixed << std::setprecision(3) << (t0 + t01) / (t1 + t2) << std::endl;
    Mat standard_sift_keypoints;
    drawKeypoints(img0, keypoints00, standard_sift_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // 保存图像
    imwrite("../output/standard_sift_keypoints.jpg", standard_sift_keypoints);
    */
    /*------------------------------------------test2: 特征匹配例子------------------------------------------*/
    string change_model = string("affine");// 注: 图像变换方式: 还可以是perspective similarity
	cv::Mat image_1 = imread("/work/home/acmhsiv3ds/SIFT_PARALLEL-sift-parallel/RSIR-SIFT/SAR-SIFT-opencv249/assets/SAR-SIFT_1.JPG", -1);
	cv::Mat image_2 = imread("/work/home/acmhsiv3ds/SIFT_PARALLEL-sift-parallel/RSIR-SIFT/SAR-SIFT-opencv249/assets/SAR-SIFT_2.JPG", -1);

	if (!image_1.data || !image_2.data){
		cout << "图像数据加载失败！" << endl;
		return;
	}

    double total_beg = (double)getTickCount();//总时间开始
    // 构建Sar_sift对象
	// int nums_1 = image_1.rows*image_1.cols;
	// int nums_2 = image_2.rows*image_2.cols;
	// int nFeatures_1 = cvRound((double)nums_1*0.008);
	// int nFeatures_2 = cvRound((double)nums_2*0.008);
	Sar_sift sar_sift_1(0, 8, 2, pow(2, 1.0 / 3.0), 0.8/5, 0.04);
	Sar_sift sar_sift_2(0, 8, 2, pow(2, 1.0 / 3.0), 0.8/5, 0.04);
    cv::Ptr<cv::SIFT> standard_sift = cv::SIFT::create();
    //参考图像特征点检测与描述
	vector<cv::KeyPoint> keypoints_1;
	vector<cv::Mat> sar_harris_fun_1, amplit_1, orient_1;
	double detect1_beg = (double)cv::getTickCount();
	sar_sift_1.detect_keys(image_1, keypoints_1, sar_harris_fun_1, amplit_1, orient_1);
	double detect1_time = ((double)cv::getTickCount() - detect1_beg) / cv::getTickFrequency();
	std::cout << "sar_sift_1 image1 detect time:" << detect1_time << "s, keypoints_1:" << keypoints_1.size() << std::endl;

	cv::Mat descriptors_1;
	double des1_beg = (double)cv::getTickCount();
	sar_sift_1.comput_des(keypoints_1, amplit_1, orient_1, descriptors_1);
	double des1_time = ((double)cv::getTickCount() - des1_beg) / cv::getTickFrequency();
	std::cout << "sar_sift_1 image1 compute time:" << des1_time << "s" << std::endl;
	// std::cout << ""

    std::vector<cv::KeyPoint> keypoints_10;
    cv::Mat descriptors_10;
    double t1_0 = (double)cv::getTickCount();
    standard_sift->detect(image_1, keypoints_10, cv::noArray());
    t1_0 = ((double)cv::getTickCount() - t1_0) / cv::getTickFrequency();
    std::cout << "Standard sift detect time: " << t1_0 << "s, kpts0 size: " << keypoints_10.size() << std::endl;
    double t1_1 = (double)cv::getTickCount();
    standard_sift->compute(image_1, keypoints_10, descriptors_10);
    // detector->detectAndCompute(img0, cv::noArray(), keypoints00, descriptors00);
    t1_1 = ((double)cv::getTickCount() - t1_1) / cv::getTickFrequency();
    std::cout << "Standard sift compute time: " << t1_1 << "s" << std::endl;

	//待配准图像特征点检测与描述
	vector<cv::KeyPoint> keypoints_2;
	vector<cv::Mat> sar_harris_fun_2, amplit_2, orient_2;
	double detect2_beg = (double)cv::getTickCount();
	sar_sift_2.detect_keys(image_2, keypoints_2, sar_harris_fun_2, amplit_2, orient_2);
	double detect2_time = ((double)cv::getTickCount() - detect2_beg) / cv::getTickFrequency();
	std::cout << "待配准图像特征点检测花费时间是： " << detect2_time << "s" << std::endl;
	std::cout << "待配准图像检测特征点个数是： " << keypoints_2.size() << std::endl;

	double des2_beg = (double)cv::getTickCount();
	cv::Mat descriptors_2;
	sar_sift_2.comput_des(keypoints_2, amplit_2, orient_2, descriptors_2);
	double des2_time = ((double)cv::getTickCount() - des2_beg) / cv::getTickFrequency();
	std::cout << "待配准图像特征点描述花费时间是： " << des2_time << "s" << std::endl;

	std::vector<cv::KeyPoint> keypoints_20;
    cv::Mat descriptors_20;
    double t2_0 = (double)cv::getTickCount();
    standard_sift->detect(image_2, keypoints_20, cv::noArray());
    t2_0 = ((double)cv::getTickCount() - t2_0) / cv::getTickFrequency();
    std::cout << "Standard sift detect time: " << t2_0 << "s, kpts0 size: " << keypoints_20.size() << std::endl;
    double t2_1 = (double)cv::getTickCount();
    standard_sift->compute(image_2, keypoints_20, descriptors_20);
    // detector->detectAndCompute(img0, cv::noArray(), keypoints00, descriptors00);
    t2_1 = ((double)cv::getTickCount() - t2_1) / cv::getTickFrequency();
    std::cout << "Standard sift compute time: " << t2_1 << "s" << std::endl;


	//描述子匹配
	double match_beg = (double)cv::getTickCount();
	//Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher();
	Ptr<cv::DescriptorMatcher> matcher = new BFMatcher(NORM_L2);
	vector<vector<cv::DMatch>> dmatchs;
	match_des(descriptors_1, descriptors_2, dmatchs, COS);
	//matcher->knnMatch(descriptors_1, descriptors_2, dmatchs,2);

	vector<cv::DMatch> right_matchs;
	Mat matched_line;
	Mat homography = match(image_1, image_2, dmatchs, keypoints_1, keypoints_2, change_model, right_matchs, matched_line);
	string str_1;
	stringstream ss;
	ss << right_matchs.size();
	ss >> str_1;
	// namedWindow(string("正确匹配特征点连线图: ") + str_1, WINDOW_AUTOSIZE);
	// imshow(string("正确匹配特征点连线图: ") + str_1, matched_line);
	double match_time = ((double)cv::getTickCount() - match_beg) / cv::getTickFrequency();
	imwrite("../output/正确匹配特征点连线图.jpg", matched_line);

	// 使用标准keypoints进行匹配的结果:
	double standard_match_beg = (double)cv::getTickCount();
	Ptr<cv::DescriptorMatcher> standard_matcher = new FlannBasedMatcher();
	// Ptr<cv::DescriptorMatcher> standard_matcher = new BFMatcher(NORM_L2);
	vector<vector<cv::DMatch>> standard_dmatchs;
	standard_matcher->knnMatch(descriptors_10, descriptors_20, standard_dmatchs,2);

	vector<cv::DMatch> standard_right_matchs;
	Mat standard_matched_line;
	Mat standard_homography = match(image_1, image_2, standard_dmatchs, keypoints_10, keypoints_20, change_model, standard_right_matchs, standard_matched_line);
	string standard_str_1;
	stringstream standard_ss;
	standard_ss << standard_right_matchs.size();
	standard_ss >> standard_str_1;

	// knn
	// standard_matcher->knnMatch(descriptors_10, descriptors_20, standard_dmatchs, 2);
    // // 筛选匹配结果
    // std::vector<DMatch> good_matches;
    // for (int i = 0; i < standard_dmatchs.size(); i++)
    // {
    //     if (standard_dmatchs[i][0].distance < 0.7 * standard_dmatchs[i][1].distance)
    //     {
    //         good_matches.push_back(standard_dmatchs[i][0]);
    //     }
    // }

    // // 绘制匹配结果
    // Mat img_matches;
    // drawMatches(image_1, keypoints_10, image_2, keypoints_20, good_matches, img_matches);

	// namedWindow(string("正确匹配特征点连线图: ") + str_1, WINDOW_AUTOSIZE);
	// imshow(string("正确匹配特征点连线图: ") + str_1, matched_line);
	double standard_match_time = ((double)cv::getTickCount() - standard_match_beg) / cv::getTickFrequency();
	imwrite("../output/标准keypoints正确匹配特征点连线图.jpg", standard_matched_line);


	std::cout << "特征点匹配阶段花费时间是： " << match_time << "s" << std::endl;
	std::cout << "待配准图像到参考图像的" << change_model << "变换矩阵是：" << std::endl;
	std::cout << homography << endl;

	double total_time = ((double)cv::getTickCount() - total_beg) / cv::getTickFrequency();
	std::cout << "总花费时间是： " << total_time << std::endl;

	//获得参考图像和待配准图像检测特征点在各层的分布个数
	vector<int> keys1_num(SAR_SIFT_LATERS), keys2_num(SAR_SIFT_LATERS);
	for (int i = 0; i < SAR_SIFT_LATERS; ++i)
	{
		keys1_num[i] = 0;//清零
		keys2_num[i] = 0;
	}
	for (size_t i = 0; i < keypoints_1.size(); ++i)
	{
		++keys1_num[keypoints_1[i].octave];
	}
	for (size_t i = 0; i < keypoints_2.size(); ++i)
	{
		++keys2_num[keypoints_2[i].octave];
	}

	//获得正确匹配点在尺度空间各层的分布
	vector<int> right_nums1(SAR_SIFT_LATERS), right_nums2(SAR_SIFT_LATERS);
	for (int i = 0; i < SAR_SIFT_LATERS; ++i)
	{
		right_nums1[i] = 0;//清零
		right_nums2[i] = 0;
	}
	for (size_t i = 0; i < right_matchs.size(); ++i)
	{
		++right_nums1[keypoints_1[right_matchs[i].queryIdx].octave];
		++right_nums2[keypoints_2[right_matchs[i].trainIdx].octave];
	}

	//把正确匹配点坐标和所在的组
	ofstream ofile("../output/position.txt");
	if (!ofile.is_open())
	{
		cout << "文件输出错误！" << endl;
		return;
	}
	ofile << "序号" << "   " << "参考坐标" << "   " <<"层号"<<"   "<<"强度"<<"    "
		<<"待配准坐标" <<"  "<<"层号"<<"  "<<"强度"<<endl;
	for (size_t i = 0; i < right_matchs.size(); ++i)
	{
		ofile << i << "->" << keypoints_1[right_matchs[i].queryIdx].pt << "    "
			<< keypoints_1[right_matchs[i].queryIdx].octave << "    "
			<< keypoints_1[right_matchs[i].queryIdx].response << "    "
			<< keypoints_2[right_matchs[i].trainIdx].pt << "    "
			<< keypoints_2[right_matchs[i].trainIdx].octave << "   "
			<< keypoints_2[right_matchs[i].trainIdx].response << endl;
	}

	ofile << "-------------------------------------------------------" << endl;
	ofile << "组号" << " " << "参考点数" << " " << "待配准点数" << " " << "参考正确数" << " " << "待配准正确数" << endl;
	for (int i = 0; i < SAR_SIFT_LATERS; ++i)
	{
		ofile << i<< "       " << keys1_num[i] << "        " << keys2_num[i] << "        " 
			<<right_nums1[i] << "        " << right_nums2[i] << endl;
	}
	
	//图像融合
	Mat fusion_image, mosaic_image, matched_image;
	image_fusion(image_1, image_2, homography, fusion_image, mosaic_image);
	// namedWindow("融合后的图像", WINDOW_AUTOSIZE);
	// imshow("融合后的图像", fusion_image);
	imwrite("../output/融合后的图像.jpg", fusion_image);
	// namedWindow("融合镶嵌后的图像", WINDOW_AUTOSIZE);
	// imshow("融合镶嵌后的图像", mosaic_image);
	imwrite("../output/融合镶嵌后的图像.jpg", mosaic_image);

}