
#include <stdio.h>

__global__ void print_from_gpu(void){
    printf("Hello Wordl! from thread [%d,%d,%d] From device\n", blockIdx.x, blockIdx.y, blockIdx.z);
}

int main(void){
    printf("Hello World form host!\n");
    dim3 Dg(3,2,3);
    print_from_gpu<<<Dg,1>>>();
    printf("host!\n");
    cudaDeviceSynchronize();
    return 0;
}