
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LENGTH 12

__global__ void MatrixMulCuda(int* M, int* N, int* P, int DimX){
    int tid, tx, ty;
    tx = blockDim.x * blockIdx.x + threadIdx.x;
    ty = blockDim.y * blockIdx.y + threadIdx.y;
    tid = DimX * ty + tx;

    int Value = 0; int MVal = 0; int NVal = 0;

    for(int i = 0; i<DimX;i++){
        MVal = M[ty * DimX + i];
        NVal = N[i * DimX + tx];
        Value += MVal * NVal;
    }
    P[tid] = Value;
}

void MatrixMul(int* M, int* N, int* P){
    int row = 0; int col = 0;
    for(row = 0; row < LENGTH; row++){
        for(col = 0; col < LENGTH; col++){
            int Destindex = row * LENGTH + col;
            for(int index = 0; index<LENGTH; index++){
                P[Destindex] += M[row * LENGTH + index] * N[col + index * LENGTH];
            }
            
        }
    }
}

void printResult(int* M, int* N, int* P){
    int row = 0; int col = 0;
    for(row = 0; row<LENGTH; row++){
        for(col = 0; col<LENGTH; col++){
            int Destindex = row * LENGTH + col;
            printf("%d ", P[Destindex]);
        }
        printf("\n");
    }
}

int main(){
    srand(time(NULL));

    const int MatrixWidth = LENGTH; const int MatrixHeight = LENGTH;
    const int MatrixSize = MatrixWidth * MatrixHeight;
    const int BufferSize = MatrixSize * sizeof(int);

    int* M; int* N; int* P_cuda; int* P_C;

    M = (int*)malloc(BufferSize);
    N = (int*)malloc(BufferSize);
    P_cuda = (int*)malloc(BufferSize);
    P_C = (int*)malloc(BufferSize);

    for(int i = 0;i< MatrixSize;i++){
        M[i] = rand()%4; N[i] = rand()%8; P_cuda[i] = 0; P_C[i] = 0;
    }

    int* dev_M; int* dev_N; int* dev_P;

    cudaMalloc((void**)&dev_M, BufferSize);
    cudaMalloc((void**)&dev_N, BufferSize);
    cudaMalloc((void**)&dev_P, BufferSize);

    cudaMemcpy(dev_M, M, BufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N, N, BufferSize, cudaMemcpyHostToDevice);

    dim3 Dg(3,3,1);
    dim3 Db(4,4,1);
    int DimX = Dg.x * Db.x;

    MatrixMulCuda<<<Dg,Db>>>(dev_M, dev_N, dev_P, DimX);
    cudaMemcpy(P_cuda, dev_P, BufferSize, cudaMemcpyDeviceToHost);
    printf("[Cuda] \n");
    printResult(M, N, P_cuda);

    printf("\n");
    printf("\n");

    printf("[CPU] \n");
    MatrixMul(M,N,P_C);
    printResult(M,N,P_C);

    cudaFree(dev_M);cudaFree(dev_N);cudaFree(dev_P);
    free(M); free(N); free(P_cuda); free(P_C);

    return 0;
}