
#include <stdio.h>
#include <sys/time.h>

void MatrixMul(int* M, int* N, int* P, int LENGTH){
    int row = 0; int col = 0;
    for(row = 0; row < LENGTH; row++){
        for(col = 0; col < LENGTH; col++){
            int Destindex = row * LENGTH + col;
            for(int index = 0; index < LENGTH; index++){
                P[Destindex] += M[row * LENGTH + index] * N[col + index * LENGTH];
            }
        }
    }
}

void printResult(int* M, int* N, int* P, int LENGTH){
    int row = 0; int col = 0;
    for(row = 0; row < LENGTH; row++){
        for(col = 0; col < LENGTH; col++){
            int Destindex = row * LENGTH + col;
            printf("%d", P[Destindex]);
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

int main(){
    srand(time(NULL));
    struct timeval cpu_start, cpu_end;

    int MatrixWidth; int MatrixHeight; int MatrixSize; int BufferSize;
    int* M; int* N; int* P_C;

    for(int LENGTH = 8; LENGTH<10001; LENGTH+=8){
        MatrixWidth = LENGTH;
        MatrixHeight = LENGTH;
        MatrixSize = MatrixWidth * MatrixHeight;
        BufferSize = MatrixSize * sizeof(int);

        M = (int*)malloc(BufferSize);
        N = (int*)malloc(BufferSize);
        P_C = (int*)malloc(BufferSize);

        for(int i = 0;i<MatrixSize;i++){
            M[i] = rand()%4; N[i] = rand()%8; P_C[i] = 0;
        }

        gettimeofday(&cpu_start,NULL);
        MatrixMul(M,N,P_C,LENGTH);
        gettimeofday(&cpu_end, NULL);

        struct timeval cpu_gap;
        getGapTime(&cpu_start, &cpu_end, &cpu_gap);
        float f_cpu_gap = timevalToFloat(&cpu_gap);

        printf("LENGTH = %d, CPU time = %.6f \n", LENGTH, f_cpu_gap);

        free(M); free(N); free(P_C);
    }
    return 0;
}