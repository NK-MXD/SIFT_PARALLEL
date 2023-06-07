extern void test_sar_sift();
extern void test_sar_sift_omp();

int main(int argc, char *argv[])
{
	// test_sar_sift();
	test_sar_sift_omp();
	//argv[0]=sar_sift.exe   argv[1]=参考图像，  argv[2]=待配准图像   argv[3]=变换类型
	// if (argc < 4){
	// 	cout << "输入参数数量不足！" << endl;
	// 	return -1;
	// }

	// if (string(argv[3]) != string("similarity") && string(argv[3]) != string("affine")
	// 	&& string(argv[3]) != string("perspective")){
	return 0;
}