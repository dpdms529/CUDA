
#include <stdio.h>
#define BLOCK_SIZE 128

__global__ void race(int*result){
    __shared__ int my_shared_variable;
    my_shared_variable = threadIdx.x;
    
    __syncthreads();
    result[threadIdx.x] = my_shared_variable;
}

int main(){
    const int size = BLOCK_SIZE;
    const int bufferSize = size*sizeof(int);

    int* result;
    result = (int*)malloc(bufferSize);
    
    int i = 0;
    for(i = 0; i < size; i++){
        result[i] = 0;
    }

    int* dev_result;
    cudaMalloc((void**)&dev_result, bufferSize);
    race<<<1,BLOCK_SIZE>>>(dev_result);
    cudaMemcpy(result, dev_result, bufferSize, cudaMemcpyDeviceToHost);
    
    for(i = 0; i < size; i++){
        printf(" result[%d] : %d\n",i,result[i]);
    }
    
    cudaFree(dev_result);
    free(result);
    
    return 0;
}