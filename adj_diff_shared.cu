#include <stdio.h>
#define BLOCK_SIZE 1024
__global__ void adj_diff_shared(int*result, int*input)
{
  int tx = threadIdx.x;
  __shared__ int s_data[BLOCK_SIZE];
  
  // each thread reads one element to s_data
  unsigned int i = blockDim.x * blockIdx.x + tx;
  s_data[tx] = input[i];
  
  // avoid race condition: ensure all loads
  // complete before continuing
  __syncthreads();
  if(tx > 0) result[i] = s_data[tx] + s_data[tx-1];
  else if(i > 0) result[i] = s_data[tx] + input[i-1];
}

int main()
{
  const int size = 1024;
  const int bufferSize = size*sizeof(int);
 
  int* result; int* input;
  result = (int*)malloc(bufferSize);
  input = (int*)malloc(bufferSize);
  int i = 0;
  for(i = 0; i < size; i++)
  {
    result[i] = 0; input[i] = i;
  }
  int* dev_result; int* dev_input;
  
  cudaMalloc((void**)&dev_result, bufferSize);
  cudaMalloc((void**)&dev_input, bufferSize);
  cudaMemcpy(dev_input, input, bufferSize, cudaMemcpyHostToDevice);
  adj_diff_shared<<<1,1024>>>(dev_result, dev_input);
  cudaMemcpy(result, dev_result, bufferSize, cudaMemcpyDeviceToHost);
  
  for(i = 0; i < 5; i++)
  {
    printf(" input[%d] : %d\n",i,input[i]);
  }
  for(i = 0; i < 5; i++)
  {
    printf(" result[%d] : %d\n",i,result[i]);
  }
  printf(" ......\n");
  for(i = size-5; i < size; i++)
  {
    printf(" input[%d] : %d\n",i,input[i]);
  }
  for(i = size-5; i < size; i++)
  {
    printf(" result[%d] : %d\n",i,result[i]);
  }
  
  cudaFree(dev_result); cudaFree(dev_input);
  free(result); free(input);
  return 0;
}