
#include <stdio.h>
#include <sys/time.h>
#define TILE_WIDTH 16

__global__ void matrixTranspose(int* M, int* MT, int DimX){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y*blockDim.y + ty;
    int col = blockIdx.x*blockDim.x + tx;
    int tid = DimX * row + col;

    __shared__ int s_m[TILE_WIDTH][TILE_WIDTH];
    
    s_m[ty][tx] = M[col*DimX + row];
    __syncthreads();
    MT[tid] = s_m[ty][tx];
}

void printResult(int* M, int* MT, int LENGTH){
    int row, col;
    printf("M\n");
    for(row = 0; row < LENGTH; row++){
        if(row%TILE_WIDTH == 0) printf("\n");
        for(col = 0; col < LENGTH; col++){
            if(col%TILE_WIDTH == 0) printf(" ");
            int DestIndex = row * LENGTH + col;
            printf("%d ", M[DestIndex]);
        }
        printf("\n");
    }

    printf("\n");
    printf("MT\n");
    for(row = 0; row < LENGTH; row++){
        if(row%TILE_WIDTH == 0) printf("\n");
        for(col = 0; col < LENGTH; col++){
            if(col%TILE_WIDTH == 0) printf(" ");
            int DestIndex = row * LENGTH + col;
            printf("%d ", MT[DestIndex]);
        }
        printf("\n");
    }
    printf("\n");
}

void getGapTime(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time){
    gap_time->tv_sec = end_time->tv_sec - start_time->tv_sec;
    gap_time->tv_usec = end_time->tv_usec - start_time->tv_usec;
    if(gap_time->tv_usec < 0){
        gap_time->tv_usec = gap_time->tv_usec + 1000000;
        gap_time->tv_sec -= 1;
    }
}

float timevalToFloat(struct timeval* time){
    double val;
    val = time->tv_sec;
    val += (time->tv_usec * 0.000001);
    return val;
}

int main(){
    srand(time(NULL));
    struct timeval htod_start, htod_end, htod_gap;
    struct timeval gpu_start, gpu_end, gpu_gap;
    struct timeval dtoh_start, dtoh_end, dtoh_gap;

    int WIDTH = 6400;
    int MatrixSize = WIDTH * WIDTH;
    int* M; int* MT;
    int* dev_M; int* dev_MT;
    int block_dim = TILE_WIDTH;
    int grid_dim = WIDTH/block_dim;

    M = (int *)malloc(MatrixSize*sizeof(int));
    MT = (int *)malloc(MatrixSize*sizeof(int));

    for(int i = 0;i<MatrixSize;i++){
        M[i] = rand()%8;
        MT[i] = 0;
    }

    cudaMalloc((void**)&dev_M, MatrixSize*sizeof(int));
    cudaMalloc((void**)&dev_MT, MatrixSize*sizeof(int));

    gettimeofday(&htod_start,NULL);
    cudaMemcpy(dev_M, M, MatrixSize*sizeof(int), cudaMemcpyHostToDevice);
    gettimeofday(&htod_end,NULL);
    getGapTime(&htod_start, &htod_end, &htod_gap);

    dim3 Dg(grid_dim, grid_dim, 1);
    dim3 Db(block_dim, block_dim, 1);
    int DimX = Dg.x * Db.x;

    gettimeofday(&gpu_start,NULL);
    matrixTranspose <<<Dg,Db>>> (dev_M,dev_MT,DimX);
    cudaDeviceSynchronize();
    gettimeofday(&gpu_end,NULL);
    getGapTime(&gpu_start, &gpu_end, &gpu_gap);

    gettimeofday(&dtoh_start, NULL);
    cudaMemcpy(MT,dev_MT, MatrixSize*sizeof(int), cudaMemcpyDeviceToHost);
    gettimeofday(&dtoh_end, NULL);
    getGapTime(&dtoh_start, &dtoh_end, &dtoh_gap);
    
    //printResult(M,MT,WIDTH);

    float f_htod_gap = timevalToFloat(&htod_gap);
    float f_gpu_gap = timevalToFloat(&gpu_gap);
    float f_dtoh_gap = timevalToFloat(&dtoh_gap);
    float total_gap = f_htod_gap + f_gpu_gap + f_dtoh_gap;

    printf("total time = %.6f, htod time = %.6f, GPU time = %6f, dtoh time = %.6f\n", total_gap, f_htod_gap, f_gpu_gap, f_dtoh_gap);

    cudaFree(dev_M); cudaFree(dev_MT);
    free(M); free(MT);
    return 0;
}