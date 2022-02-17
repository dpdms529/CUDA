#include <stdio.h>
__global__ void adj_diff_global(int*result, int*input)
{
  // compute this thread's global index
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i > 0)
  {
    // each thread loads two elements from global memory
    int x_i = input[i];
    int x_i_minus_one = input[i-1];
   
    // compute the difference using values stored in registers
    result[i] = x_i + x_i_minus_one;
  }
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
  adj_diff_global<<<1,1024>>>(dev_result, dev_input);
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