
#include <stdio.h>
#define MAX_SHAREDSIZE 2048
__global__ void LoadStoreSharedMemory(int*In, int*Out){
    int LoadStoreSize = MAX_SHAREDSIZE/blockDim.x;
    
    int begin = LoadStoreSize*threadIdx.x;
    int end = begin+LoadStoreSize;
    
    __shared__ int SharedMemory[MAX_SHAREDSIZE];
    
    int i = 0;
    for(i = begin; i < end; i++)
      SharedMemory[i] = In[i];
    
    __syncthreads();
    
    for(i = begin; i < end; i++)
      Out[i] = SharedMemory[i];
    
    __syncthreads();
}

int main(){
    const int size = MAX_SHAREDSIZE;
    const int BufferSize = size*sizeof(int);

    int* Input; int* Output;
    
    Input = (int*)malloc( BufferSize);
    Output = (int*)malloc( BufferSize);
    
    int i = 0;
    
    for(i = 0; i < size; i++) {
        Input[i] = i; Output[i] = 0;
    }
    
    int* dev_In; int* dev_Out;
    
    cudaMalloc((void**)&dev_In, size*sizeof(int));
    cudaMalloc((void**)&dev_Out, size*sizeof(int));
    
    cudaMemcpy(dev_In, Input, size*sizeof(int), cudaMemcpyHostToDevice);
    
    LoadStoreSharedMemory<<<32,512>>>(dev_In, dev_Out);
    
    cudaMemcpy(Output, dev_Out, size*sizeof(int), cudaMemcpyDeviceToHost);
    
    for(i = 0; i < 5; i++){
        printf(" Output[%d] : %d\n",i,Output[i]);
    }
    printf(" ......\n");
    for(i = size-5; i < size; i++){
        printf(" Output[%d] : %d\n",i,Output[i]);
    }
    
    cudaFree(dev_In); cudaFree(dev_Out);
    free(Input); free(Output);
    
    return 0;
}