
#include <stdio.h>

__global__ void add(int *a, int *b, int *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

int main(void){
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int arr_cnt;

    printf("type the array count : ");
    scanf("%d", &arr_cnt);

    a = (int*)malloc(arr_cnt * sizeof(int));
    b = (int*)malloc(arr_cnt * sizeof(int));
    c = (int*)malloc(arr_cnt * sizeof(int));

    cudaMalloc((void**)&dev_a, arr_cnt * sizeof(int));
    cudaMalloc((void**)&dev_b, arr_cnt * sizeof(int));
    cudaMalloc((void**)&dev_c, arr_cnt * sizeof(int));
    
    for(int i = 0;i<arr_cnt;i++){
        a[i] = i; b[i] = i;
    }

    cudaMemcpy(dev_a, a, arr_cnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arr_cnt * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 1024;
    int div = arr_cnt/threads_per_block;
    printf("arr_cnt/threads_per_block = %d \n", div);
    add <<<div + 1, threads_per_block>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, arr_cnt * sizeof(int), cudaMemcpyDeviceToHost);

    bool success = true;
    for(int i = 0;i<arr_cnt;i++){
        if((a[i] + b[i] != c[i])){
            printf("Error : %d + %d != %d\n", a[i],b[i],c[i]);
            success = false;
        }
    }
    if(success) printf("We did it!\n");

    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    free(a); free(b); free(c);

    return 0;
}