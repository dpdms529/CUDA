
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#define TILEWID 8

__global__ void kmeans(int k, int DimX, float * x, float * y, float * centroidX, float * centroidY, int * cluster, int * allSame, float * clusterSumX, float * clusterSumY, int * clusterSize) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = DimX * row + col;

    float minDis;                               //각 centroid까지의 거리 중 최소 거리
    int temp;                                   //이전 cluster
    __shared__ int isSame;                      //같은 블럭내에서 이전 cluster와 새로 할당받은 clsuter가 같은 데이터 수
    __shared__ float s_x[TILEWID][TILEWID];     //x 데이터를 저장하는 shared memory
    __shared__ float s_y[TILEWID][TILEWID];     //y 데이터를 저장하는 shared memory
    __shared__ int s_cluster[TILEWID][TILEWID]; //cluster 데이터를 저장하는 shared memory
    
    //초기화
    temp = cluster[tid];
    minDis = FLT_MAX;
    s_x[ty][tx] = x[tid];
    s_y[ty][tx] = y[tid];
    __syncthreads();
    
    //스레드 인덱스가 (0,0)일 때 isSame 값 초기화
    if(tx == 0 && ty == 0) isSame = 0;
    __syncthreads();

    //데이터 좌표에서 각 centroid까지의 거리 중 최소 거리 구하고, 최소 거리에 있는 centroid가 속한 cluster로 할당
    for(int i = 0;i<k;i++){
        float xdis = pow(centroidX[i] - s_x[ty][tx],2);
        float ydis = pow(centroidY[i] - s_y[ty][tx],2);
        float dis = sqrt(xdis + ydis);
        if(minDis>dis) {
            minDis = dis;
            s_cluster[ty][tx] = i;
            __syncthreads();
        }
    }
    cluster[tid] = s_cluster[ty][tx];
    
    //할당받은 cluster가 이전 cluster와 같으면 isSame 1증가
    if(s_cluster[ty][tx] == temp) atomicAdd(&isSame, 1);
    __syncthreads();

    //스레드 인덱스가 (0,0)일 때
    if(tx == 0 && ty == 0) {
        //allSame에 각 블록 별로 구해진 isSame값 더하기
        atomicAdd(allSame, isSame);
        __syncthreads();

        //각 클러스터에 할당된 데이터들의 평균 지점 구하고, 평균 지점으로 centroid 이동
        for(int i = 0;i<k;i++){
            float sumX = 0;
            float sumY = 0;
            int count = 0;
            for(int r = 0;r<TILEWID;r++){
                for(int c = 0;c<TILEWID;c++){
                    //각 블록 별로 각 클러스터에 할당된 데이터 값 더하고, 카운트 세기
                    if(s_cluster[r][c] == i){ 
                        sumX += s_x[r][c];
                        sumY += s_y[r][c];
                        count++;
                        __syncthreads();
                    }        
                } 
            }
            //블록 별로 구해진 클러스터 별 데이터 합과 카운트 값 합치기
            atomicAdd(&clusterSumX[i], sumX);
            atomicAdd(&clusterSumY[i], sumY);
            atomicAdd(&clusterSize[i], count);
            __syncthreads();
            
            //클러스터 별 평균 구하기
            centroidX[i] = clusterSumX[i]/clusterSize[i];
            centroidY[i] = clusterSumY[i]/clusterSize[i];
            __syncthreads();
            
        } 
    }
    
}

void printResult(int k, int length, float * x, float * y, float * centroidX, float * centroidY, int * cluster){
    printf("\ndata\n");
    for(int i = 0;i<length;i++){
        printf("x[%d] : %f\t y[%d] : %f\n", i, x[i], i, y[i]);
    }
    
    printf("\nresult\n");
    for(int i = 0;i<k;i++){
         printf("centroidX[%d] : %f\t centroidY[%d] : %f\n", i, centroidX[i], i, centroidY[i]);
         for(int j = 0;j<length;j++){
             if(cluster[j] == i) printf("x[%d] : %f\t y[%d] : %f\n", j, x[j], j, y[j]);
         }
         printf("\n");
    }
    
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

int main() {
    srand(time(NULL));
    struct timeval htod_start, htod_end, htod_gap;
    struct timeval gpu_start, gpu_end, gpu_gap;
    struct timeval dtoh_start, dtoh_end, dtoh_gap;

    int length;   //데이터 개수
    int k = 3;    //클러스터 개수

    int block_dim = TILEWID;    //block 차원수
    int grid_dim;               //grid 차원수
    
    float * x; float * y; float * centroidX; float * centroidY; int * cluster; int allSame;

    float * d_x; float * d_y; float * d_centroidX; float * d_centroidY; int * d_cluster; int * d_allSame;
    float * d_clusterSumX; float * d_clusterSumY; int * d_clusterSize;

    for(int wid = TILEWID; wid<=TILEWID*30 ;wid += TILEWID){
        length = wid * wid;         
        grid_dim = wid/block_dim; 

        //host 메모리 할당
        x = (float *)malloc(length * sizeof(float));    //데이터 x좌표
        y = (float *)malloc(length * sizeof(float));    //데이터 y좌표
        centroidX = (float *)malloc(k * sizeof(float)); //centroid x좌표
        centroidY = (float *)malloc(k * sizeof(float)); //centroid y좌표
        cluster = (int *)malloc(length * sizeof(int));  //각 데이터가 할당받은 cluster
        
        //데이터 생성
        for(int i = 0;i<length;i++){
            x[i] = rand()%100;
            y[i] = rand()%100;
        }

        //초기 centroid 선택(데이터 중 앞에 있는 k개)
        for(int i = 0;i<k;i++){
            centroidX[i] = x[i];
            centroidY[i] = y[i];
        }

        //각 데이터 별 clsuter 초기화
        for(int i = 0;i<length;i++){
                cluster[i] = -1;
        }

        //device 메모리 할당
        cudaMalloc((void **)&d_x, length * sizeof(float));
        cudaMalloc((void **)&d_y, length * sizeof(float));
        cudaMalloc((void **)&d_centroidX, k * sizeof(float));
        cudaMalloc((void **)&d_centroidY, k * sizeof(float));
        cudaMalloc((void **)&d_cluster, length * sizeof(int));
        cudaMalloc((void **)&d_allSame, sizeof(int));
        cudaMalloc((void **)&d_clusterSumX, k * sizeof(float));
        cudaMalloc((void **)&d_clusterSumY, k * sizeof(float));
        cudaMalloc((void **)&d_clusterSize, k * sizeof(int));

        //host to device 메모리 복사
        gettimeofday(&htod_start, NULL);
        cudaMemcpy(d_x, x, length * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, length * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_centroidX, centroidX, k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_centroidY, centroidY, k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster, cluster, length * sizeof(int), cudaMemcpyHostToDevice);
        gettimeofday(&htod_end, NULL);
        getGapTime(&htod_start, &htod_end, &htod_gap);

        //스레드 개수 설정
        dim3 Dg(grid_dim, grid_dim, 1);
        dim3 Db(block_dim, block_dim, 1);
        int DimX = Dg.x * Db.x;

        //cluster 구하기
        gettimeofday(&gpu_start, NULL);
        while(1){
            cudaMemset(d_allSame, 0, sizeof(int));
            cudaMemset(d_clusterSumX, 0, k*sizeof(float));
            cudaMemset(d_clusterSumY, 0, k*sizeof(float));
            cudaMemset(d_clusterSize, 0, k*sizeof(int));
            kmeans<<<Dg,Db>>>(k, DimX, d_x, d_y, d_centroidX, d_centroidY, d_cluster, d_allSame, d_clusterSumX, d_clusterSumY, d_clusterSize);
            cudaMemcpy(&allSame, d_allSame, sizeof(int), cudaMemcpyDeviceToHost);
            if(allSame == length) break;        //모든 데이터의 이전에 할당받은 cluster와 새로 할당받은 clsuter가 같으면 clustering 종료
        }
        gettimeofday(&gpu_end, NULL);
        getGapTime(&gpu_start, &gpu_end, &gpu_gap);
        
        //device to host 메모리 복사
        gettimeofday(&dtoh_start, NULL);
        cudaMemcpy(centroidX, d_centroidX, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(centroidY, d_centroidY, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cluster, d_cluster, length * sizeof(int), cudaMemcpyDeviceToHost);
        gettimeofday(&dtoh_end, NULL);
        getGapTime(&dtoh_start, &dtoh_end, &dtoh_gap);

        //결과 출력
        //printResult(k, length, x, y, centroidX, centroidY, cluster);
        float f_htod_gap = timevalToFloat(&htod_gap);
        float f_gpu_gap = timevalToFloat(&gpu_gap);
        float f_dtoh_gap = timevalToFloat(&dtoh_gap);
        float total_gap = f_htod_gap + f_gpu_gap + f_dtoh_gap;
        printf("\nlength = %d\t total time = %.6f\t htod timd = %.6f\t GPU time = %.6f\t dtoh time = %.6f\n", length, total_gap, f_htod_gap, f_gpu_gap, f_dtoh_gap);

        //메모리 해제
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_centroidX); cudaFree(d_centroidY); cudaFree(d_cluster); cudaFree(d_allSame); 
        cudaFree(d_clusterSumX); cudaFree(d_clusterSumY); cudaFree(d_clusterSize);
        free(x); free(y); free(centroidX); free(centroidY); free(cluster);

    }
    return 0;
}