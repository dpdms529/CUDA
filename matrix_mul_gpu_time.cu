#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define TILE_WIDTH 16

__global__ void MatrixMulCudaShared( int*M, int*N, int*P, int DimX)
{
  int tid, tx, ty;
  tx = blockDim.x * blockIdx.x + threadIdx.x;
  ty = blockDim.y * blockIdx.y + threadIdx.y;
  tid = DimX * ty + tx;
  
  __shared__ int s_a[TILE_WIDTH][TILE_WIDTH];
  __shared__ int s_b[TILE_WIDTH][TILE_WIDTH];
  
  int Value = 0;
  for (int i = 0; i < DimX/TILE_WIDTH; ++i)
  {
    s_a[ty][tx] = M[ty*DimX + (i*TILE_WIDTH+tx)];
    s_b[ty][tx] = N[(i*TILE_WIDTH+ty)*DimX + tx];
    __syncthreads();
    for (int k=0; k<TILE_WIDTH; ++k)
      Value += s_a[ty][k] * s_b[k][tx];
    __syncthreads();
  }
  P[tid] = Value;
}

__global__ void MatrixMulCuda( int*M, int*N, int*P, int DimX)
{
  int tid, tx, ty;
  tx = blockDim.x * blockIdx.x + threadIdx.x;
  ty = blockDim.y * blockIdx.y + threadIdx.y;
  tid = DimX * ty + tx;
  
  int Value = 0; int MVal = 0; int NVal = 0;
  
  for (int i = 0; i < DimX; i++)
  {
    MVal = M[ty * DimX + i];
    NVal = N[i * DimX + tx];
    Value += MVal * NVal;
  }
  P[tid] = Value;
}

void printResult(int* M, int* N, int* P, int LENGTH)
{
  int row = 0; int col = 0;
  for (row = 0; row < LENGTH; row++)
  {
    for (col = 0; col < LENGTH; col++)
    {
      int Destindex = row * LENGTH + col;
      printf( "%d ", P[Destindex]);
    }
    printf( "\n");
  }
}

void getGapTime(struct timeval* start_time, struct timeval* end_time, struct timeval* gap_time)
{
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

int main()
{
  srand(time(NULL));
  struct timeval htod_start, htod_end;
  struct timeval gpu_start, gpu_end;
  struct timeval dtoh_start, dtoh_end;
 
  int* M; int* N; int* P_cuda;
  int MatrixWidth; int MatrixHeight; int MatrixSize; int BufferSize;
  
  int* dev_M; int* dev_N; int* dev_P;
  
  int grid_dim;
  int block_dim=TILE_WIDTH;
  
  for(int LENGTH=TILE_WIDTH; LENGTH<TILE_WIDTH*1000; LENGTH+=TILE_WIDTH)
  {
    grid_dim = LENGTH / block_dim;
    
    MatrixWidth = LENGTH;
    MatrixHeight = LENGTH;
    MatrixSize = MatrixWidth * MatrixHeight;
    BufferSize = MatrixSize * sizeof(int);
    
    M = (int*)malloc(BufferSize);
    N = (int*)malloc(BufferSize);
    P_cuda = (int*)malloc(BufferSize);
  
    for( int i = 0; i < MatrixSize; i++)
    {
      M[i] = rand()%4; N[i] = rand()%8; P_cuda[i] = 0;
    }
    
    cudaMalloc((void**)&dev_M, BufferSize);
    cudaMalloc((void**)&dev_N, BufferSize);
    cudaMalloc((void**)&dev_P, BufferSize);
    
    gettimeofday(&htod_start, NULL);
    cudaMemcpy(dev_M, M, BufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N, N, BufferSize, cudaMemcpyHostToDevice);
    gettimeofday(&htod_end, NULL);
    struct timeval htod_gap;
    getGapTime(&htod_start, &htod_end, &htod_gap);
    
    dim3 Dg(grid_dim, grid_dim, 1);
    dim3 Db(block_dim, block_dim, 1);
    int DimX = Dg.x * Db.x;

    gettimeofday(&gpu_start, NULL);
    // MatrixMulCuda <<<Dg,Db>>> (dev_M, dev_N, dev_P, DimX);
    MatrixMulCudaShared <<<Dg,Db>>> (dev_M, dev_N, dev_P, DimX);
    cudaDeviceSynchronize();
    gettimeofday(&gpu_end, NULL);
    
    gettimeofday(&dtoh_start, NULL);
    cudaMemcpy(P_cuda, dev_P, BufferSize, cudaMemcpyDeviceToHost);
    gettimeofday(&dtoh_end, NULL);
    
    struct timeval gpu_gap, dtoh_gap;
    getGapTime(&gpu_start, &gpu_end, &gpu_gap);
    getGapTime(&dtoh_start, &dtoh_end, &dtoh_gap);
    
    float f_htod_gap = timevalToFloat(&htod_gap);
    float f_gpu_gap = timevalToFloat(&gpu_gap);
    float f_dtoh_gap = timevalToFloat(&dtoh_gap);
    float total_gap = f_htod_gap + f_gpu_gap + f_dtoh_gap;
    printf("[Cuda] LENGTH = %d, total time = %.6f, htod time = %.6f, GPU time = %.6f, dtoh time = %.6f \n",
           LENGTH, total_gap, f_htod_gap, f_gpu_gap, f_dtoh_gap);
    
    cudaFree(dev_M); cudaFree(dev_N); cudaFree(dev_P);
    free(M); free(N); free(P_cuda);
  }
  return 0;
}