
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

void kmeans(int k, int length, float * x, float * y, float * centroidX, float * centroidY, int * cluster, int * allSame) {
    float minDis[length];   //각 centroid까지의 거리 중 최소 거리
    int temp[length];       //이전 cluster

    //초기화
    for(int i = 0;i<length;i++){
        temp[i] = cluster[i];
        minDis[i] = FLT_MAX;
    }

    //데이터 좌표에서 각 centroid까지의 거리 중 최소 거리 구하고, 최소 거리에 있는 centroid가 속한 cluster로 할당
    for(int i = 0;i<k;i++){
        for(int j = 0;j<length;j++){
            float xdis = pow(centroidX[i] - x[j],2);
            float ydis = pow(centroidY[i] - y[j],2);
            float dis = sqrt(xdis + ydis);
            if(minDis[j]>dis) {
                minDis[j] = dis;
                cluster[j] = i;
            }
        }
    }
 
    //할당받은 cluster가 이전 cluster와 같으면 allSame 1증가
    for(int i = 0;i<length;i++){
        if(cluster[i] == temp[i]) *allSame = *allSame + 1;
    }
    
    //각 클러스터에 할당된 데이터들의 평균 지점 구하고, 평균 지점으로 centroid 이동
    for(int i = 0; i<k; i++){
        float sumX = 0;
        float sumY = 0;
        int count = 0;
        for(int j = 0;j<length;j++){
            if(cluster[j] == i){
                sumX += x[j];
                sumY += y[j];
                count++;
            }
        }
        centroidX[i] = sumX/count;
        centroidY[i] = sumY/count;
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
    struct timeval cpu_start, cpu_end, cpu_gap;

    int length;   //데이터 개수
    int k = 3;    //클러스터 개수
    
    float *x; float *y; float *centroidX; float *centroidY; int *cluster; int allSame;

    for(int wid = 8; wid<=240; wid+=8){
        length = wid*wid;   

        //메모리 할당
        x = (float *)malloc(length * sizeof(float));      //데이터 x좌표
        y = (float *)malloc(length * sizeof(float));      //데이터 y좌표
        centroidX = (float *)malloc(k * sizeof(float));   //centroid x좌표
        centroidY = (float *)malloc(k * sizeof(float));   //centroid y좌표
        cluster = (int *)malloc(length * sizeof(int));    //각 데이터가 할당받은 cluster

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

        //cluster 구하기
        gettimeofday(&cpu_start,NULL);
        while(1){
            allSame = 0;
            kmeans(k, length, x, y, centroidX, centroidY, cluster, &allSame);
            if(allSame == length) break;  //모든 데이터의 이전에 할당받은 cluster와 새로 할당받은 clsuter가 같으면 clustering 종료
        }
        gettimeofday(&cpu_end,NULL);
        getGapTime(&cpu_start, &cpu_end, &cpu_gap);
        
        //결과 출력
        float f_cpu_gap = timevalToFloat(&cpu_gap);
        //printResult(k, length, x, y, centroidX, centroidY, cluster);
        printf("\nlength = %d\t CPU time = %.6f\n", length, f_cpu_gap);

        //메모리 해제
        free(x); free(y); free(centroidX); free(centroidY); free(cluster);
    }

    return 0;
}