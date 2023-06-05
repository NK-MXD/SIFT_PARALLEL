extern void test_match();
extern void test_omp();
extern void test_multithreading();
extern void mtest_multithreadandSIMD();

int main(int argc, char *argv[])
{
//    test_match();
//    test_omp();
    // test_multithreading();
    mtest_multithreadandSIMD();
    return 0;
}