//extern void test_match();
//extern void test_omp();
//extern void test_multithreading();
//extern void mtest_multithreadandSIMD();
//extern void test_opencv_para();
//extern void mtest_simd();
//extern void mtest_multithreadandSIMD();
extern void test_scalability();

int main(int argc, char *argv[])
{
//    test_match();
//    test_omp();
    // test_multithreading();
    // mtest_multithreadandSIMD();
//    test_multithreading();
    // test_opencv_para();
    // mtest_simd();
//    mtest_multithreadandSIMD();
    test_scalability();
    return 0;
}