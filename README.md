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
