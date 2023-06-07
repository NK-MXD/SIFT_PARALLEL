import matplotlib.pyplot as plt


with open('input.txt', 'r') as f:
    img_h = []
    img_w = []
    proc = []
    serial_detect = []
    serial_compute = []
    serial_kpts = []
    opencv_detect = []
    opencv_compute = []
    opencv_kpts = []
    omp_detect = []
    omp_compute = []
    omp_kpts = []
    omp_simd_detect = []
    omp_simd_compute = []
    omp_simd_kpts = []
    serial_total = []
    opencv_total = []
    omp_total = []
    omp_simd_total = []
    for i in range(4):
        img_h.append(int(f.readline()))
        img_w.append(int(f.readline()))
        proc.append(int(f.readline()))
        serial_detect.append(float(f.readline()))
        serial_compute.append(float(f.readline()))
        serial_kpts.append(int(f.readline()))
        opencv_detect.append(float(f.readline()))
        opencv_compute.append(float(f.readline()))
        opencv_kpts.append(int(f.readline()))
        omp_detect.append(float(f.readline()))
        omp_compute.append(float(f.readline()))
        omp_kpts.append(int(f.readline()))
        omp_simd_detect.append(float(f.readline()))
        omp_simd_compute.append(float(f.readline()))
        omp_simd_kpts.append(int(f.readline()))

        serial_total.append(serial_detect[i] + serial_compute[i])
        opencv_total.append(opencv_detect[i] + opencv_compute[i])
        omp_total.append(omp_detect[i] + omp_compute[i])
        omp_simd_total.append(omp_simd_detect[i] + omp_simd_compute[i])

    # draw a plt that x is num of processors, y is time
    plt.figure(1)
    plt.plot(proc, serial_detect, 'r', label='serial')
    plt.plot(proc, opencv_detect, 'g', label='opencv')
    plt.plot(proc, omp_detect, 'b', label='omp')
    plt.plot(proc, omp_simd_detect, 'y', label='omp_simd')
    plt.xlabel('num of processors')
    plt.ylabel('time')
    plt.title('detect time')
    plt.legend()
    plt.savefig('assets/detect.png')

    plt.figure(2)
    plt.plot(proc, serial_compute, 'r', label='serial')
    plt.plot(proc, opencv_compute, 'g', label='opencv')
    plt.plot(proc, omp_compute, 'b', label='omp')
    plt.plot(proc, omp_simd_compute, 'y', label='omp_simd')
    plt.xlabel('num of processors')
    plt.ylabel('time')
    plt.title('compute time')
    plt.legend()
    plt.savefig('assets/compute.png')

    plt.figure(3)
    plt.plot(proc, serial_total, 'r', label='serial')
    plt.plot(proc, opencv_total, 'g', label='opencv')
    plt.plot(proc, omp_total, 'b', label='omp')
    plt.plot(proc, omp_simd_total, 'y', label='omp_simd')
    plt.xlabel('num of processors')
    plt.ylabel('time')
    plt.title('total time')
    plt.legend()
    plt.savefig('assets/total.png')

    plt.figure(4)
    plt.plot(proc, serial_kpts, 'r', label='serial')
    plt.plot(proc, opencv_kpts, 'g', label='opencv')
    plt.plot(proc, omp_kpts, 'b', label='omp')
    plt.plot(proc, omp_simd_kpts, 'y', label='omp_simd')
    plt.xlabel('num of processors')
    plt.ylabel('num of keypoints')
    plt.title('num of keypoints')
    plt.legend()
    plt.savefig('assets/kpts.png')



