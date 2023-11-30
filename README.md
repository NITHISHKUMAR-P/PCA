# PCA
## Exp-1 GPU based vector summation
```cu
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}
```
## Exp-2 Matrix summation with a 2D grid and 2D blocks
```cu
void sumMatrixOnHost(int *A, int *B, int *C, const int nx, const int ny) {

int *ia = A; int *ib = B; int *ic = C;

for (int iy = 0; iy < ny; iy++)
{
    for (int ix = 0; ix < nx; ix++)
    {
        ic[ix] = ia[ix] + ib[ix];

    }

    ia += nx;
    ib += nx;
    ic += nx;
}

return;
}
```
## Exp-3 Simple warp divergence
```cu
__global__ void reduceUnrolling8 (int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceUnrolling16 (int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 16;

    if (idx + 15 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        int c1 = g_idata[idx + 8 * blockDim.x];
        int c2 = g_idata[idx + 9 * blockDim.x];
        int c3 = g_idata[idx + 10 * blockDim.x];
        int c4 = g_idata[idx + 11 * blockDim.x];
        int d1 = g_idata[idx + 12 * blockDim.x];
        int d2 = g_idata[idx + 13 * blockDim.x];
        int d3 = g_idata[idx + 14 * blockDim.x];
        int d4 = g_idata[idx + 15 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 + c1 + c2 + c3 + c4
                       + d1 + d2 + d3 + d4;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
```
## Exp-4 Matrix addition with unified memory
```cu
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}
```
## Exp-5 Matrix maltiplication using cuda c
```cu
__global__ void matrixMultiply(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int k = 0; k < size; ++k)
    {
        sum += a[row * size + k] * b[k * size + col];
    }
    c[row * size + col] = sum;
}
```
## Exp-6 Demonstrate Matrix transposition on shared memory
```cu
__global__ void setRowReadRow(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x] ;
}

__global__ void setColReadCol(int *out)
{
    __shared__ int tile[BDIMX][BDIMY];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadCol2(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;
    tile[icol][irow] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow] ;
}

__global__ void setRowReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColPad(int *out)
{
    __shared__ int tile[BDIMY][BDIMX + IPAD];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow] ;
}

__global__ void setRowReadColDyn(int *out)
{
    extern  __shared__ int tile[];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;
    unsigned int col_idx = icol * blockDim.x + irow;
    tile[idx] = idx;
    __syncthreads();
    out[idx] = tile[col_idx];
}

__global__ void setRowReadColDynPad(int *out)
{
    extern  __shared__ int tile[];
    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = g_idx / blockDim.y;
    unsigned int icol = g_idx % blockDim.y;
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;
    tile[row_idx] = g_idx;
    __syncthreads();
    out[g_idx] = tile[col_idx];
}
```
