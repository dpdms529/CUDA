
#include <stdio.h>

#define LENGTH 14

__global__ void MatrixAdd(int* M, int* N, int* P, int DimX){
    int tid, tx, ty;
    tx = blockDim.x * blockIdx.x + threadIdx.x;
    ty = blockDim.y * blockIdx.y + threadIdx.y;
    tid = DimX * ty + tx;

    P[tid] = M[tid] + N[tid];
}

void printResult(int* M, int* N, int* P){
    int row = 0; int col = 0;
    for(row = 0;row<LENGTH;row++){
        for(col = 0;col<LENGTH;col++){
            int Destindex = row*LENGTH + col;
            printf("%d (= C[%d][%d]) = %d (= A[%d][%d]) + %d (= B[%d][%d]) \n",
                   P[Destindex], row, col, M[Destindex], row, col, N[Destindex],row, col);
        }
    }
}

int main(){
    const int MatrixWidth = LENGTH; const int MatrixHeight = LENGTH;
    const int MatrixSize = MatrixWidth * MatrixHeight;
    const int BufferSize = MatrixSize * sizeof(int);

    int* M; int* N; int* P_cuda;
    M = (int*)malloc(BufferSize);
    N = (int*)malloc(BufferSize);
    P_cuda = (int*)malloc(BufferSize);

    for(int i = 0;i< MatrixSize;i++){
        M[i] = i; N[i] = i; P_cuda[i] = 0;
    }

    int* dev_M; int* dev_N; int* dev_P;

    cudaMalloc((void**)&dev_M, BufferSize);
    cudaMalloc((void**)&dev_N, BufferSize);
    cudaMalloc((void**)&dev_P, BufferSize);

    cudaMemcpy(dev_M, M, BufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N, N, BufferSize, cudaMemcpyHostToDevice);

    dim3 Dg(3,3,1);
    dim3 Db(8,6,1);
    int DimX = 3 * 8;

    MatrixAdd<<<Dg,Db>>>(dev_M, dev_N, dev_P, DimX);
    cudaMemcpy(P_cuda, dev_P, BufferSize, cudaMemcpyDeviceToHost);

    printResult(M,N,P_cuda);

    cudaFree(dev_M); cudaFree(dev_N); cudaFree(dev_P);
    free(M); free(N); free(P_cuda);

    return 0;
}