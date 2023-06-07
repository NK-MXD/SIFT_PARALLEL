#include "sift_cuda.cuh"

/* kernel configuration */

__constant__ int d_MaxNumPoints;
__device__ unsigned int d_PointCounter[8 * 2 + 1];
__constant__ float d_ScaleDownKernel[5];
__constant__ float d_LowPassKernel[2 * LOWPASS_R + 1];
__constant__ float d_LaplaceKernel[8 * 12 * 16];

/* kernel functions */

__global__ void RescalePositionsKernel(SiftPoint *d_sift, int numPts, float scale) {
    int num = blockIdx.x * blockDim.x + threadIdx.x;
    if (num < numPts) {
        d_sift[num].xpos *= scale;
        d_sift[num].ypos *= scale;
        d_sift[num].scale *= scale;
    }
}

__global__ void ScaleUpKernel(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int x = blockIdx.x * SCALEUP_W + 2 * tx;
    int y = blockIdx.y * SCALEUP_H + 2 * ty;
    if (x < 2 * width && y < 2 * height) {
        int xl = blockIdx.x * (SCALEUP_W / 2) + tx;
        int yu = blockIdx.y * (SCALEUP_H / 2) + ty;
        int xr = min(xl + 1, width - 1);
        int yd = min(yu + 1, height - 1);
        float vul = d_Data[yu * pitch + xl];
        float vur = d_Data[yu * pitch + xr];
        float vdl = d_Data[yd * pitch + xl];
        float vdr = d_Data[yd * pitch + xr];
        d_Result[(y + 0) * newpitch + x + 0] = vul;
        d_Result[(y + 0) * newpitch + x + 1] = 0.50f * (vul + vur);
        d_Result[(y + 1) * newpitch + x + 0] = 0.50f * (vul + vdl);
        d_Result[(y + 1) * newpitch + x + 1] = 0.25f * (vul + vur + vdl + vdr);
    }
}

__global__ void ScaleDownKernel(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch) {
    __shared__ float inrow[SCALEDOWN_W + 4];
    __shared__ float brow[5 * (SCALEDOWN_W / 2)];
    __shared__ int yRead[SCALEDOWN_H + 4];
    __shared__ int yWrite[SCALEDOWN_H + 4];
#define dx2 (SCALEDOWN_W/2)
    const int tx = threadIdx.x;
    const int tx0 = tx + 0 * dx2;
    const int tx1 = tx + 1 * dx2;
    const int tx2 = tx + 2 * dx2;
    const int tx3 = tx + 3 * dx2;
    const int tx4 = tx + 4 * dx2;
    const int xStart = blockIdx.x * SCALEDOWN_W;
    const int yStart = blockIdx.y * SCALEDOWN_H;
    const int xWrite = xStart / 2 + tx;
    float k0 = d_ScaleDownKernel[0];
    float k1 = d_ScaleDownKernel[1];
    float k2 = d_ScaleDownKernel[2];
    if (tx < SCALEDOWN_H + 4) {
        int y = yStart + tx - 2;
        y = (y < 0 ? 0 : y);
        y = (y >= height ? height - 1 : y);
        yRead[tx] = y * pitch;
        yWrite[tx] = (yStart + tx - 4) / 2 * newpitch;
    }
    __syncthreads();
    int xRead = xStart + tx - 2;
    xRead = (xRead < 0 ? 0 : xRead);
    xRead = (xRead >= width ? width - 1 : xRead);

    int maxtx = min(dx2, width / 2 - xStart / 2);
    for (int dy = 0; dy < SCALEDOWN_H + 4; dy += 5) {
        {
            inrow[tx] = d_Data[yRead[dy + 0] + xRead];
            __syncthreads();
            if (tx < maxtx) {
                brow[tx4] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) +
                            k2 * inrow[2 * tx + 2];
                if (dy >= 4 && !(dy & 1))
                    d_Result[yWrite[dy + 0] + xWrite] =
                            k2 * brow[tx2] + k0 * (brow[tx0] + brow[tx4]) + k1 * (brow[tx1] + brow[tx3]);
            }
            __syncthreads();
        }
        if (dy < (SCALEDOWN_H + 3)) {
            inrow[tx] = d_Data[yRead[dy + 1] + xRead];
            __syncthreads();
            if (tx < maxtx) {
                brow[tx0] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) +
                            k2 * inrow[2 * tx + 2];
                if (dy >= 3 && (dy & 1))
                    d_Result[yWrite[dy + 1] + xWrite] =
                            k2 * brow[tx3] + k0 * (brow[tx1] + brow[tx0]) + k1 * (brow[tx2] + brow[tx4]);
            }
            __syncthreads();
        }
        if (dy < (SCALEDOWN_H + 2)) {
            inrow[tx] = d_Data[yRead[dy + 2] + xRead];
            __syncthreads();
            if (tx < maxtx) {
                brow[tx1] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) +
                            k2 * inrow[2 * tx + 2];
                if (dy >= 2 && !(dy & 1))
                    d_Result[yWrite[dy + 2] + xWrite] =
                            k2 * brow[tx4] + k0 * (brow[tx2] + brow[tx1]) + k1 * (brow[tx3] + brow[tx0]);
            }
            __syncthreads();
        }
        if (dy < (SCALEDOWN_H + 1)) {
            inrow[tx] = d_Data[yRead[dy + 3] + xRead];
            __syncthreads();
            if (tx < maxtx) {
                brow[tx2] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) +
                            k2 * inrow[2 * tx + 2];
                if (dy >= 1 && (dy & 1))
                    d_Result[yWrite[dy + 3] + xWrite] =
                            k2 * brow[tx0] + k0 * (brow[tx3] + brow[tx2]) + k1 * (brow[tx4] + brow[tx1]);
            }
            __syncthreads();
        }
        if (dy < SCALEDOWN_H) {
            inrow[tx] = d_Data[yRead[dy + 4] + xRead];
            __syncthreads();
            if (tx < dx2 && xWrite < width / 2) {
                brow[tx3] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) +
                            k2 * inrow[2 * tx + 2];
                if (!(dy & 1))
                    d_Result[yWrite[dy + 4] + xWrite] =
                            k2 * brow[tx1] + k0 * (brow[tx4] + brow[tx3]) + k1 * (brow[tx0] + brow[tx2]);
            }
            __syncthreads();
        }
    }
}

__global__ void LowPassKernel(float *d_Image, float *d_Result, int width, int pitch, int height) {
    __shared__ float xrows[16][32];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int xp = blockIdx.x * LOWPASS_W + tx;
    const int yp = blockIdx.y * LOWPASS_H + ty;
    const int N = 16;
    float *k = d_LowPassKernel;
    int xl = max(min(xp - 4, width - 1), 0);
#pragma unroll
    for (int l = -8; l < 4; l += 4) {
        int ly = l + ty;
        int yl = max(min(yp + l + 4, height - 1), 0);
        float val = d_Image[yl * pitch + xl];
        val = k[4] * ShiftDown(val, 4) +
              k[3] * (ShiftDown(val, 5) + ShiftDown(val, 3)) +
              k[2] * (ShiftDown(val, 6) + ShiftDown(val, 2)) +
              k[1] * (ShiftDown(val, 7) + ShiftDown(val, 1)) +
              k[0] * (ShiftDown(val, 8) + val);
        xrows[ly + 8][tx] = val;
    }
    __syncthreads();
#pragma unroll
    for (int l = 4; l < LOWPASS_H; l += 4) {
        int ly = l + ty;
        int yl = min(yp + l + 4, height - 1);
        float val = d_Image[yl * pitch + xl];
        val = k[4] * ShiftDown(val, 4) +
              k[3] * (ShiftDown(val, 5) + ShiftDown(val, 3)) +
              k[2] * (ShiftDown(val, 6) + ShiftDown(val, 2)) +
              k[1] * (ShiftDown(val, 7) + ShiftDown(val, 1)) +
              k[0] * (ShiftDown(val, 8) + val);
        xrows[(ly + 8) % N][tx] = val;
        int ys = yp + l - 4;
        if (xp < width && ys < height && tx < LOWPASS_W)
            d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
                                        k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
                                        k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
                                        k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
                                        k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
        __syncthreads();
    }
    int ly = LOWPASS_H + ty;
    int ys = yp + LOWPASS_H - 4;
    if (xp < width && ys < height && tx < LOWPASS_W)
        d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
                                    k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
                                    k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
                                    k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
                                    k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
}

__device__ float FastAtan2Kernel(float y, float x) {
    float absx = abs(x);
    float absy = abs(y);
    float a = __fdiv_rn(min(absx, absy), max(absx, absy));
    float s = a * a;
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    r = (absy > absx ? 1.57079637f - r : r);
    r = (x < 0 ? 3.14159274f - r : r);
    r = (y < 0 ? -r : r);
    return r;
}

__global__
void LaplaceMultiKernel(float *d_Image, float *d_Result, int width, int pitch, int height, int octave) {
    __shared__ float buff[(LAPLACE_W + 2 * LAPLACE_R) * LAPLACE_S];
    const int tx = threadIdx.x;
    const int xp = blockIdx.x * LAPLACE_W + tx;
    const int yp = blockIdx.y;
    float *data = d_Image + max(min(xp - LAPLACE_R, width - 1), 0);
    float temp[2 * LAPLACE_R + 1], kern[LAPLACE_S][LAPLACE_R + 1];
    if (xp < (width + 2 * LAPLACE_R)) {
        for (int i = 0; i <= 2 * LAPLACE_R; i++)
            temp[i] = data[max(0, min(yp + i - LAPLACE_R, height - 1)) * pitch];
        for (int scale = 0; scale < LAPLACE_S; scale++) {
            float *buf = buff + (LAPLACE_W + 2 * LAPLACE_R) * scale;
            float *kernel = d_LaplaceKernel + octave * 12 * 16 + scale * 16;
            for (int i = 0; i <= LAPLACE_R; i++)
                kern[scale][i] = kernel[i];
            float sum = kern[scale][0] * temp[LAPLACE_R];
#pragma unroll
            for (int j = 1; j <= LAPLACE_R; j++)
                sum += kern[scale][j] * (temp[LAPLACE_R - j] + temp[LAPLACE_R + j]);
            buf[tx] = sum;
        }
    }
    __syncthreads();
    if (tx < LAPLACE_W && xp < width) {
        int scale = 0;
        float oldRes = kern[scale][0] * buff[tx + LAPLACE_R];
#pragma unroll
        for (int j = 1; j <= LAPLACE_R; j++)
            oldRes += kern[scale][j] * (buff[tx + LAPLACE_R - j] + buff[tx + LAPLACE_R + j]);
        for (int scale = 1; scale < LAPLACE_S; scale++) {
            float *buf = buff + (LAPLACE_W + 2 * LAPLACE_R) * scale;
            float res = kern[scale][0] * buf[tx + LAPLACE_R];
#pragma unroll
            for (int j = 1; j <= LAPLACE_R; j++)
                res += kern[scale][j] * (buf[tx + LAPLACE_R - j] + buf[tx + LAPLACE_R + j]);
            d_Result[(scale - 1) * height * pitch + yp * pitch + xp] = res - oldRes;
            oldRes = res;
        }
    }
}

__global__
void FindPointsMultiKernel(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling,
                   float lowestScale, float thresh, float factor, float edgeLimit, int octave) {
#define MEMWID (MINMAX_W + 2)
    __shared__ unsigned short points[2 * MEMWID];

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        atomicMax(&d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
        atomicMax(&d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave - 1]);
    }
    int tx = threadIdx.x;
    int block = blockIdx.x / NUM_SCALES;
    int scale = blockIdx.x - NUM_SCALES * block;
    int minx = block * MINMAX_W;
    int maxx = min(minx + MINMAX_W, width);
    int xpos = minx + tx;
    int size = pitch * height;
    int ptr = size * scale + max(min(xpos - 1, width - 1), 0);

    int yloops = min(height - MINMAX_H * blockIdx.y, MINMAX_H);
    float maxv = 0.0f;
    for (int y = 0; y < yloops; y++) {
        int ypos = MINMAX_H * blockIdx.y + y;
        int yptr1 = ptr + ypos * pitch;
        float val = d_Data0[yptr1 + 1 * size];
        maxv = fmaxf(maxv, fabs(val));
    }
    if (!__any_sync(0xffffffff, maxv > thresh))
        return;

    int ptbits = 0;
    for (int y = 0; y < yloops; y++) {

        int ypos = MINMAX_H * blockIdx.y + y;
        int yptr1 = ptr + ypos * pitch;
        float d11 = d_Data0[yptr1 + 1 * size];
        if (__any_sync(0xffffffff, fabs(d11) > thresh)) {

            int yptr0 = ptr + max(0, ypos - 1) * pitch;
            int yptr2 = ptr + min(height - 1, ypos + 1) * pitch;
            float d01 = d_Data0[yptr1];
            float d10 = d_Data0[yptr0 + 1 * size];
            float d12 = d_Data0[yptr2 + 1 * size];
            float d21 = d_Data0[yptr1 + 2 * size];

            float d00 = d_Data0[yptr0];
            float d02 = d_Data0[yptr2];
            float ymin1 = fminf(fminf(d00, d01), d02);
            float ymax1 = fmaxf(fmaxf(d00, d01), d02);
            float d20 = d_Data0[yptr0 + 2 * size];
            float d22 = d_Data0[yptr2 + 2 * size];
            float ymin3 = fminf(fminf(d20, d21), d22);
            float ymax3 = fmaxf(fmaxf(d20, d21), d22);
            float ymin2 = fminf(fminf(ymin1, fminf(fminf(d10, d12), d11)), ymin3);
            float ymax2 = fmaxf(fmaxf(ymax1, fmaxf(fmaxf(d10, d12), d11)), ymax3);

            float nmin2 = fminf(ShiftUp(ymin2, 1), ShiftDown(ymin2, 1));
            float nmax2 = fmaxf(ShiftUp(ymax2, 1), ShiftDown(ymax2, 1));
            float minv = fminf(fminf(nmin2, ymin1), ymin3);
            minv = fminf(fminf(minv, d10), d12);
            float maxv = fmaxf(fmaxf(nmax2, ymax1), ymax3);
            maxv = fmaxf(fmaxf(maxv, d10), d12);

            if (tx > 0 && tx < MINMAX_W + 1 && xpos <= maxx)
                ptbits |= ((d11 < fminf(-thresh, minv)) | (d11 > fmaxf(thresh, maxv))) << y;
        }
    }

    unsigned int totbits = __popc(ptbits);
    unsigned int numbits = totbits;
    for (int d = 1; d < 32; d <<= 1) {
        unsigned int num = ShiftUp(totbits, d);
        if (tx >= d)
            totbits += num;
    }
    int pos = totbits - numbits;
    for (int y = 0; y < yloops; y++) {
        int ypos = MINMAX_H * blockIdx.y + y;
        if (ptbits & (1 << y) && pos < MEMWID) {
            points[2 * pos + 0] = xpos - 1;
            points[2 * pos + 1] = ypos;
            pos++;
        }
    }

    totbits = Shuffle(totbits, 31);
    if (tx < totbits) {
        int xpos = points[2 * tx + 0];
        int ypos = points[2 * tx + 1];
        int ptr = xpos + (ypos + (scale + 1) * height) * pitch;
        float val = d_Data0[ptr];
        float *data1 = &d_Data0[ptr];
        float dxx = 2.0f * val - data1[-1] - data1[1];
        float dyy = 2.0f * val - data1[-pitch] - data1[pitch];
        float dxy = 0.25f * (data1[+pitch + 1] + data1[-pitch - 1] - data1[-pitch + 1] - data1[+pitch - 1]);
        float tra = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;
        if (tra * tra < edgeLimit * det) {
            float edge = __fdividef(tra * tra, det);
            float dx = 0.5f * (data1[1] - data1[-1]);
            float dy = 0.5f * (data1[pitch] - data1[-pitch]);
            float *data0 = d_Data0 + ptr - height * pitch;
            float *data2 = d_Data0 + ptr + height * pitch;
            float ds = 0.5f * (data0[0] - data2[0]);
            float dss = 2.0f * val - data2[0] - data0[0];
            float dxs = 0.25f * (data2[1] + data0[-1] - data0[1] - data2[-1]);
            float dys = 0.25f * (data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
            float idxx = dyy * dss - dys * dys;
            float idxy = dys * dxs - dxy * dss;
            float idxs = dxy * dys - dyy * dxs;
            float idet = __fdividef(1.0f, idxx * dxx + idxy * dxy + idxs * dxs);
            float idyy = dxx * dss - dxs * dxs;
            float idys = dxy * dxs - dxx * dys;
            float idss = dxx * dyy - dxy * dxy;
            float pdx = idet * (idxx * dx + idxy * dy + idxs * ds);
            float pdy = idet * (idxy * dx + idyy * dy + idys * ds);
            float pds = idet * (idxs * dx + idys * dy + idss * ds);
            if (pdx < -0.5f || pdx > 0.5f || pdy < -0.5f || pdy > 0.5f || pds < -0.5f || pds > 0.5f) {
                pdx = __fdividef(dx, dxx);
                pdy = __fdividef(dy, dyy);
                pds = __fdividef(ds, dss);
            }
            float dval = 0.5f * (dx * pdx + dy * pdy + ds * pds);
            int maxPts = d_MaxNumPoints;
            float sc = powf(2.0f, (float) scale / NUM_SCALES) * exp2f(pds * factor);
            if (sc >= lowestScale) {
                atomicMax(&d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
                unsigned int idx = atomicInc(&d_PointCounter[2 * octave + 0], 0x7fffffff);
                idx = (idx >= maxPts ? maxPts - 1 : idx);
                d_Sift[idx].xpos = xpos + pdx;
                d_Sift[idx].ypos = ypos + pdy;
                d_Sift[idx].scale = sc;
                d_Sift[idx].sharpness = val + dval;
                d_Sift[idx].edgeness = edge;
                d_Sift[idx].subsampling = subsampling;
            }
        }
    }
}

__global__ void ComputeOrientationsKernel(cudaTextureObject_t texObj, SiftPoint *d_Sift, int octave) {
    __shared__ float hist[64];
    __shared__ float gauss[11];
    const int tx = threadIdx.x;

    int fstPts = min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
    int totPts = min(d_PointCounter[2 * octave + 0], d_MaxNumPoints);
    for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {

        float i2sigma2 = -1.0f / (2.0f * 1.5f * 1.5f * d_Sift[bx].scale * d_Sift[bx].scale);
        if (tx < 11)
            gauss[tx] = exp(i2sigma2 * (tx - 5) * (tx - 5));
        if (tx < 64)
            hist[tx] = 0.0f;
        __syncthreads();
        float xp = d_Sift[bx].xpos - 4.5f;
        float yp = d_Sift[bx].ypos - 4.5f;
        int yd = tx / 11;
        int xd = tx - yd * 11;
        float xf = xp + xd;
        float yf = yp + yd;
        if (yd < 11) {
            float dx = tex2D<float>(texObj, xf + 1.0, yf) - tex2D<float>(texObj, xf - 1.0, yf);
            float dy = tex2D<float>(texObj, xf, yf + 1.0) - tex2D<float>(texObj, xf, yf - 1.0);
            int bin = 16.0f * atan2f(dy, dx) / 3.1416f + 16.5f;
            if (bin > 31)
                bin = 0;
            float grad = sqrtf(dx * dx + dy * dy);
            atomicAdd(&hist[bin], grad * gauss[xd] * gauss[yd]);
        }
        __syncthreads();
        int x1m = (tx >= 1 ? tx - 1 : tx + 31);
        int x1p = (tx <= 30 ? tx + 1 : tx - 31);
        if (tx < 32) {
            int x2m = (tx >= 2 ? tx - 2 : tx + 30);
            int x2p = (tx <= 29 ? tx + 2 : tx - 30);
            hist[tx + 32] = 6.0f * hist[tx] + 4.0f * (hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
        }
        __syncthreads();
        if (tx < 32) {
            float v = hist[32 + tx];
            hist[tx] = (v > hist[32 + x1m] && v >= hist[32 + x1p] ? v : 0.0f);
        }
        __syncthreads();
        if (tx == 0) {
            float maxval1 = 0.0;
            float maxval2 = 0.0;
            int i1 = -1;
            int i2 = -1;
            for (int i = 0; i < 32; i++) {
                float v = hist[i];
                if (v > maxval1) {
                    maxval2 = maxval1;
                    maxval1 = v;
                    i2 = i1;
                    i1 = i;
                } else if (v > maxval2) {
                    maxval2 = v;
                    i2 = i;
                }
            }
            float val1 = hist[32 + ((i1 + 1) & 31)];
            float val2 = hist[32 + ((i1 + 31) & 31)];
            float peak = i1 + 0.5f * (val1 - val2) / (2.0f * maxval1 - val1 - val2);
            d_Sift[bx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
            atomicMax(&d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave + 0]);
            if (maxval2 > 0.8f * maxval1) {
                float val1 = hist[32 + ((i2 + 1) & 31)];
                float val2 = hist[32 + ((i2 + 31) & 31)];
                float peak = i2 + 0.5f * (val1 - val2) / (2.0f * maxval2 - val1 - val2);
                unsigned int idx = atomicInc(&d_PointCounter[2 * octave + 1], 0x7fffffff);
                if (idx < d_MaxNumPoints) {
                    d_Sift[idx].xpos = d_Sift[bx].xpos;
                    d_Sift[idx].ypos = d_Sift[bx].ypos;
                    d_Sift[idx].scale = d_Sift[bx].scale;
                    d_Sift[idx].sharpness = d_Sift[bx].sharpness;
                    d_Sift[idx].edgeness = d_Sift[bx].edgeness;
                    d_Sift[idx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);;
                    d_Sift[idx].subsampling = d_Sift[bx].subsampling;
                }
            }
        }
        __syncthreads();
    }
}

__global__ void
ExtractSiftDescriptorsKernel(cudaTextureObject_t texObj, SiftPoint *d_sift, float subsampling, int octave) {
    __shared__ float gauss[16];
    __shared__ float buffer[128];
    __shared__ float sums[4];

    const int tx = threadIdx.x; // 0 -> 16
    const int ty = threadIdx.y; // 0 -> 8
    const int idx = ty * 16 + tx;
    if (ty == 0)
        gauss[tx] = __expf(-(tx - 7.5f) * (tx - 7.5f) / 128.0f);

    int fstPts = min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
    int totPts = min(d_PointCounter[2 * octave + 1], d_MaxNumPoints);
    //if (tx==0 && ty==0)
    //  printf("%d %d %d %d\n", octave, fstPts, min(d_PointCounter[2*octave], d_MaxNumPoints), totPts);
    for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {

        buffer[idx] = 0.0f;
        __syncthreads();

        // Compute angles and gradients
        float theta = 2.0f * 3.1415f / 360.0f * d_sift[bx].orientation;
        float sina = __sinf(theta);           // cosa -sina
        float cosa = __cosf(theta);           // sina  cosa
        float scale = 12.0f / 16.0f * d_sift[bx].scale;
        float ssina = scale * sina;
        float scosa = scale * cosa;

        for (int y = ty; y < 16; y += 8) {
            float xpos = d_sift[bx].xpos + (tx - 7.5f) * scosa - (y - 7.5f) * ssina + 0.5f;
            float ypos = d_sift[bx].ypos + (tx - 7.5f) * ssina + (y - 7.5f) * scosa + 0.5f;
            float dx = tex2D<float>(texObj, xpos + cosa, ypos + sina) -
                       tex2D<float>(texObj, xpos - cosa, ypos - sina);
            float dy = tex2D<float>(texObj, xpos - sina, ypos + cosa) -
                       tex2D<float>(texObj, xpos + sina, ypos - cosa);
            float grad = gauss[y] * gauss[tx] * __fsqrt_rn(dx * dx + dy * dy);
            float angf = 4.0f / 3.1415f * FastAtan2Kernel(dy, dx) + 4.0f;

            int hori = (tx + 2) / 4 - 1;      // Convert from (tx,y,angle) to bins
            float horf = (tx - 1.5f) / 4.0f - hori;
            float ihorf = 1.0f - horf;
            int veri = (y + 2) / 4 - 1;
            float verf = (y - 1.5f) / 4.0f - veri;
            float iverf = 1.0f - verf;
            int angi = angf;
            int angp = (angi < 7 ? angi + 1 : 0);
            angf -= angi;
            float iangf = 1.0f - angf;

            int hist = 8 * (4 * veri + hori);   // Each gradient measure is interpolated
            int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
            int p2 = angp + hist;
            if (tx >= 2) {
                float grad1 = ihorf * grad;
                if (y >= 2) {   // Upper left
                    float grad2 = iverf * grad1;
                    atomicAdd(buffer + p1, iangf * grad2);
                    atomicAdd(buffer + p2, angf * grad2);
                }
                if (y <= 13) {  // Lower left
                    float grad2 = verf * grad1;
                    atomicAdd(buffer + p1 + 32, iangf * grad2);
                    atomicAdd(buffer + p2 + 32, angf * grad2);
                }
            }
            if (tx <= 13) {
                float grad1 = horf * grad;
                if (y >= 2) {    // Upper right
                    float grad2 = iverf * grad1;
                    atomicAdd(buffer + p1 + 8, iangf * grad2);
                    atomicAdd(buffer + p2 + 8, angf * grad2);
                }
                if (y <= 13) {   // Lower right
                    float grad2 = verf * grad1;
                    atomicAdd(buffer + p1 + 40, iangf * grad2);
                    atomicAdd(buffer + p2 + 40, angf * grad2);
                }
            }
        }
        __syncthreads();

        // Normalize twice and suppress peaks first time
        float sum = buffer[idx] * buffer[idx];
        for (int i = 16; i > 0; i /= 2)
            sum += ShiftDown(sum, i);
        if ((idx & 31) == 0)
            sums[idx / 32] = sum;
        __syncthreads();
        float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
        tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);

        sum = tsum1 * tsum1;
        for (int i = 16; i > 0; i /= 2)
            sum += ShiftDown(sum, i);
        if ((idx & 31) == 0)
            sums[idx / 32] = sum;
        __syncthreads();

        float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
        float *desc = d_sift[bx].data;
        desc[idx] = tsum1 * rsqrtf(tsum2);
        if (idx == 0) {
            d_sift[bx].xpos *= subsampling;
            d_sift[bx].ypos *= subsampling;
            d_sift[bx].scale *= subsampling;
        }
        __syncthreads();
    }
}

/*
 * Allocate memory for scale space and difference of Gaussian
 * return: allocated memory pointer
 */
float *AllocSiftTempMemory(int width, int height, int nOctaves, bool scaleUp) {
//    TimerGPU timer(nullptr);
    const int nd = NUM_SCALES + 3;
    int w = width * (scaleUp ? 2 : 1);
    int h = height * (scaleUp ? 2 : 1);
    int p = iAlignUp(w, 128);
    int size = h * p;                 // image sizes
    int sizeTmp = nd * h * p;           // laplace buffer sizes
    for (int i = 0; i < nOctaves; i++) {
        w /= 2;
        h /= 2;
        int p = iAlignUp(w, 128);
        size += h * p;
        sizeTmp += nd * h * p;
    }
    float *memoryTmp = nullptr;
    size_t pitch;
    size += sizeTmp;
    safeCall(cudaMallocPitch((void **) &memoryTmp, &pitch, (size_t) 4096, (size + 4095) / 4096 * sizeof(float)));
//    printf("Allocated memory size: %d bytes\n", size);
//    printf("Memory allocation time =      %.2f ms\n\n", timer.read());
    return memoryTmp;
}

void FreeSiftTempMemory(float *memoryTmp) {
    if (memoryTmp) safeCall(cudaFree(memoryTmp));
}

/*
 * Extract SIFT features from an image
 * return: time taken
 */
double ExtractSift(SiftData &siftData, CudaImage &img, int nOctave, double initBlur, float thresh, float lowestScale,
                   bool scaleUp, float *tempMemory) {
    TimerGPU timer(nullptr);
    unsigned int *d_PointCounterAddr;
    safeCall(cudaGetSymbolAddress((void **) &d_PointCounterAddr, d_PointCounter));      // get address of counter
    safeCall(cudaMemset(d_PointCounterAddr, 0, (8 * 2 + 1) * sizeof(int)));        // set counter to 0
    safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)));        // copy max point number to constant memory

    // buffer layout: [init] [gpyr] [dogpyr]
    const int nd = NUM_SCALES + 3;
    int w = img.width * (scaleUp ? 2 : 1);
    int h = img.height * (scaleUp ? 2 : 1);
    int p = iAlignUp(w, 128);
    int width = w, height = h;
    int size = h * p;                 // init image sizes (pitched)
    int sizeTmp = nd * h * p;           // sum to get laplace buffer size
    for (int i = 0; i < nOctave; i++) {
        w /= 2;
        h /= 2;
        int p = iAlignUp(w, 128);
        size += h * p;
        sizeTmp += nd * h * p;
    }
    float *memoryTmp = tempMemory;
    size += sizeTmp;
    if (!tempMemory) {
        size_t pitch;
        safeCall(cudaMallocPitch((void **) &memoryTmp, &pitch, (size_t) 4096, (size + 4095) / 4096 * sizeof(float)));
    }
    float *memorySub = memoryTmp + sizeTmp;

    CudaImage lowImg;
    lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);
    if (!scaleUp) {
        float kernel[8 * 12 * 16];
        PrepareLaplaceKernels(nOctave, 0.0f, kernel);
        safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8 * 12 * 16 * sizeof(float)));
        LowPass(lowImg, img, (float) std::max(initBlur, 0.001));
        TimerGPU timer1(nullptr);
        ExtractSiftLoop(siftData, lowImg, nOctave, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height * iAlignUp(width, 128));
        safeCall(cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2 * nOctave], sizeof(int), cudaMemcpyDeviceToHost));
        siftData.numPts = (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
        printf("SIFT extraction time =        %.2f ms %d\n", timer1.read(), siftData.numPts);
    } else {
        CudaImage upImg;
        upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
        TimerGPU timer1(nullptr);
        ScaleUp(upImg, img);
        LowPass(lowImg, upImg, (float) std::max(initBlur, 0.001));
        float kernel[8 * 12 * 16];
        PrepareLaplaceKernels(nOctave, 0.0f, kernel);
        safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8 * 12 * 16 * sizeof(float)));
        ExtractSiftLoop(siftData, lowImg, nOctave, 0.0f, thresh, lowestScale * 2.0f, 1.0f, memoryTmp, memorySub + height * iAlignUp(width, 128));
        safeCall(cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2 * nOctave], sizeof(int), cudaMemcpyDeviceToHost));
        siftData.numPts = (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
        RescalePositions(siftData, 0.5f);
        printf("SIFT extraction time =        %.2f ms\n", timer1.read());
    }
    if (!tempMemory)
        safeCall(cudaFree(memoryTmp));
    if (siftData.h_data)
        safeCall(cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint) * siftData.numPts, cudaMemcpyDeviceToHost));
    double totTime = timer.read();
    printf("With memcpy =  %.2f ms %d\n\n", totTime, siftData.numPts);
    return totTime;
}

void InitSiftData(SiftData &data, int num, bool host, bool dev) {
    data.numPts = 0;
    data.maxPts = num;
    int sz = (int) sizeof(SiftPoint) * num;
    data.h_data = nullptr;
    if (host)
        data.h_data = (SiftPoint *) malloc(sz);
    data.d_data = nullptr;
    if (dev)
        safeCall(cudaMalloc((void **) &data.d_data, sz));
}

void FreeSiftData(SiftData &data) {
    if (data.d_data != nullptr)
        safeCall(cudaFree(data.d_data));
    data.d_data = nullptr;
    if (data.h_data != nullptr)
        free(data.h_data);
    data.numPts = 0;
    data.maxPts = 0;
}

/*
 * For each octave, first compute the first img of the octave, then extract sift feature.
 * return: time taken
 */
double ExtractSiftLoop(SiftData &siftData, CudaImage &img, int nOctave, double initBlur, float thresh, float lowestScale,
                       float subsampling, float *memoryTmp, float *memorySub) {
    TimerGPU timer(nullptr);
    int w = img.width;
    int h = img.height;
    if (nOctave > 1) {
        CudaImage subImg;
        int _w = w / 2;
        int _h = h / 2;
        int p = iAlignUp(_w, 128);
        subImg.Allocate(_w, _h, p, false, memorySub);
        ScaleDown(subImg, img, 0.5f);
        float totInitBlur = (float) sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
        ExtractSiftLoop(siftData, subImg, nOctave - 1, totInitBlur, thresh, lowestScale, subsampling * 2.0f, memoryTmp, memorySub + _h * p);
    }
    ExtractSiftOctave(siftData, img, nOctave, thresh, lowestScale, subsampling, memoryTmp);
    double totTime = timer.read();
//    printf("ExtractSift time total =      %.2f ms %d\n\n", totTime, nOctave);
    return totTime;
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp) {
    const int nd = NUM_SCALES + 3;
    CudaImage diffImg[nd];
    int w = img.width;
    int h = img.height;
    int p = iAlignUp(w, 128);
    for (int i = 0; i < nd - 1; i++)
        diffImg[i].Allocate(w, h, p, false, memoryTmp + i * p * h);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = img.d_data;
    resDesc.res.pitch2D.width = img.width;
    resDesc.res.pitch2D.height = img.height;
    resDesc.res.pitch2D.pitchInBytes = img.pitch * sizeof(float);
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    float baseBlur = pow(2.0f, -1.0f / NUM_SCALES);
    float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
    LaplaceMulti(texObj, img, diffImg, octave); // diffImg = img - baseBlur * img
    FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f / NUM_SCALES, lowestScale / subsampling, subsampling, octave);   // find local maxima and minima
    ComputeOrientations(texObj, img, siftData, octave); // compute orientation for each point
    ExtractSiftDescriptors(texObj, siftData, subsampling, octave);  // extract sift descriptor for each point
    //OrientAndExtract(texObj, siftData, subsampling, octave);

    safeCall(cudaDestroyTextureObject(texObj));
}

/*
 * Compute the difference of Gaussian of the image
 */
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave) {
    int width = results[0].width;
    int pitch = results[0].pitch;
    int height = results[0].height;
    dim3 threads(LAPLACE_W + 2 * LAPLACE_R);
    dim3 blocks(iDivUp(width, LAPLACE_W), height);
    LaplaceMultiKernel<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
    checkMsg("LaplaceMulti() execution failed\n");
    return 0.0;
}

double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave) {
    if (sources->d_data == nullptr) {
        printf("FindPointsMulti: missing data\n");
        return 0.0;
    }
    int w = sources->width;
    int p = sources->pitch;
    int h = sources->height;
    dim3 blocks(iDivUp(w, MINMAX_W) * NUM_SCALES, iDivUp(h, MINMAX_H));
    dim3 threads(MINMAX_W + 2);
    FindPointsMultiKernel<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
    checkMsg("FindPointsMulti() execution failed\n");
    return 0.0;
}

double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave) {
    dim3 blocks(512);
    dim3 threads(11 * 11);
    ComputeOrientationsKernel<<<blocks, threads>>>(texObj, siftData.d_data, octave);
    checkMsg("ComputeOrientations() execution failed\n");
    return 0.0;
}

double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave) {
    dim3 blocks(512);
    dim3 threads(16, 8);
    ExtractSiftDescriptorsKernel<<<blocks, threads>>>(texObj, siftData.d_data, subsampling, octave);
    checkMsg("ExtractSiftDescriptors() execution failed\n");
    return 0.0;
}

void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel) {
    if (numOctaves > 1) {
        float totInitBlur = (float) sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
        PrepareLaplaceKernels(numOctaves - 1, totInitBlur, kernel);
    }
    float scale = pow(2.0f, -1.0f / NUM_SCALES);
    float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
    for (int i = 0; i < NUM_SCALES + 3; i++) {
        float kernelSum = 0.0f;
        float var = scale * scale - initBlur * initBlur;
        for (int j = 0; j <= LAPLACE_R; j++) {
            kernel[numOctaves * 12 * 16 + 16 * i + j] = (float) expf(-(double) j * j / 2.0 / var);
            kernelSum += (j == 0 ? 1 : 2) * kernel[numOctaves * 12 * 16 + 16 * i + j];
        }
        for (int j = 0; j <= LAPLACE_R; j++)
            kernel[numOctaves * 12 * 16 + 16 * i + j] /= kernelSum;
        scale *= diffScale;
    }
}

double LowPass(CudaImage &res, CudaImage &src, float scale) {
    float kernel[2 * LOWPASS_R + 1];
    static float oldScale = -1.0f;
    if (scale != oldScale) {
        float kernelSum = 0.0f;
        float ivar2 = 1.0f / (2.0f * scale * scale);
        for (int j = -LOWPASS_R; j <= LOWPASS_R; j++) {
            kernel[j + LOWPASS_R] = (float) expf(-(double) j * j * ivar2);
            kernelSum += kernel[j + LOWPASS_R];
        }
        for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
            kernel[j + LOWPASS_R] /= kernelSum;
        safeCall(cudaMemcpyToSymbol(d_LowPassKernel, kernel, (2 * LOWPASS_R + 1) * sizeof(float)));
        oldScale = scale;
    }
    int width = res.width;
    int pitch = res.pitch;
    int height = res.height;
    dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
    dim3 threads(LOWPASS_W + 2 * LOWPASS_R, 4);
    LowPassKernel<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);
    checkMsg("LowPass() execution failed\n");
    return 0.0;
}

double ScaleUp(CudaImage &res, CudaImage &src) {
    if (res.d_data == nullptr || src.d_data == nullptr) {
        printf("ScaleUp: missing data\n");
        return 0.0;
    }
    dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
    dim3 threads(SCALEUP_W / 2, SCALEUP_H / 2);
    ScaleUpKernel<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
    checkMsg("ScaleUp() execution failed\n");
    return 0.0;
}

double ScaleDown(CudaImage &res, CudaImage &src, float variance) {
    static float oldVariance = -1.0f;
    if (res.d_data == nullptr || src.d_data == nullptr) {
        printf("ScaleDown: missing data\n");
        return 0.0;
    }
    if (oldVariance != variance) {
        float h_Kernel[5];
        float kernelSum = 0.0f;
        for (int j = 0; j < 5; j++) {
            h_Kernel[j] = (float) expf(-(double) (j - 2) * (j - 2) / 2.0 / variance);
            kernelSum += h_Kernel[j];
        }
        for (int j = 0; j < 5; j++)
            h_Kernel[j] /= kernelSum;
        safeCall(cudaMemcpyToSymbol(d_ScaleDownKernel, h_Kernel, 5 * sizeof(float)));
        oldVariance = variance;
    }
    dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
    dim3 threads(SCALEDOWN_W + 4);
    ScaleDownKernel<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
    checkMsg("ScaleDown() execution failed\n");
    return 0.0;
}

double RescalePositions(SiftData &siftData, float scale) {
    dim3 blocks(iDivUp(siftData.numPts, 64));
    dim3 threads(64);
    RescalePositionsKernel<<<blocks, threads>>>(siftData.d_data, siftData.numPts, scale);
    checkMsg("RescapePositions() execution failed\n");
    return 0.0;
}

void matchAndDraw(SiftData &siftData1, SiftData &siftData2, cv::Mat &img1, cv::Mat &img2) {
    std::vector<cv::KeyPoint> kpts1, kpts2;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> good_matches;
    cv::Mat desc1 = cv::Mat(siftData1.numPts, 128, CV_32F);
    cv::Mat desc2 = cv::Mat(siftData2.numPts, 128, CV_32F);
    cv::Mat img_kpts1, img_kpts2, img_match;
    kpts1.resize(siftData1.numPts);
    kpts2.resize(siftData2.numPts);
    for (int i = 0; i < siftData1.numPts; i++) {
        kpts1[i].pt.x = siftData1.h_data[i].xpos;
        kpts1[i].pt.y = siftData1.h_data[i].ypos;
        kpts1[i].response = siftData1.h_data[i].sharpness / 255;
        kpts1[i].size = siftData1.h_data[i].scale;
        kpts1[i].angle = siftData1.h_data[i].orientation;
        kpts1[i].octave = pow(2, 8 - sqrt(siftData1.h_data[i].subsampling));
        float *desc = desc1.ptr<float>(i);
        for (int j = 0; j < 128; j++)
            desc[j] = siftData1.h_data[i].data[j];
    }
    for (int i = 0; i < siftData2.numPts; i++) {
        kpts2[i].pt.x = siftData2.h_data[i].xpos;
        kpts2[i].pt.y = siftData2.h_data[i].ypos;
        kpts2[i].response = siftData2.h_data[i].sharpness / 255;
        kpts2[i].size = siftData2.h_data[i].scale;
        kpts2[i].angle = siftData2.h_data[i].orientation;
        kpts2[i].octave = pow(2, 8 - sqrt(siftData2.h_data[i].subsampling));
        float *desc = desc2.ptr<float>(i);
        for (int j = 0; j < 128; j++)
            desc[j] = siftData2.h_data[i].data[j];
    }
    cv::Ptr<cv::DescriptorMatcher> matcher = new cv::BFMatcher();
    matcher->knnMatch(desc1, desc2, matches, 2);
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < 0.50 * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
    cv::drawKeypoints(img1, kpts1, img_kpts1);
    cv::drawKeypoints(img2, kpts2, img_kpts2);
    cv::drawMatches(img1, kpts1, img2, kpts2, good_matches, img_match, cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));
    std::cout << "kpts1: " << kpts1.size() << ", kpts2: " << kpts2.size() << ", matches: " << matches.size()
              << ", good_matches: " << good_matches.size() << std::endl;
    cv::imwrite("kpts1.png", img_kpts1);
    cv::imwrite("kpts2.png", img_kpts2);
    cv::imwrite("match.png", img_match);
}