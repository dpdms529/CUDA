
#include <stdio.h>
#include <sys/time.h>
#define TILE_WIDTH 16

void matrixTranspose(int* M, int* MT, int LENGTH){
    int row, col;
    for(row = 0;row<LENGTH;row++){
        for(col = 0;col<LENGTH;col++){
            int DestIndex = row * LENGTH + col;
            MT[DestIndex] = M[col * LENGTH + row];
        }
    }
}

void printResult(int* M, int* MT, int LENGTH){
    int row, col;
    printf("M\n");
    for(row = 0; row < LENGTH; row++){
        if(row%TILE_WIDTH == 0) printf("\n");
        for(col = 0; col < LENGTH; col++){
            if(col%TILE_WIDTH == 0) printf(" ");
            int DestIndex = row * LENGTH + col;
            printf("%d ", M[DestIndex]);
        }
        printf("\n");
    }

    printf("\n");
    printf("MT\n");
    for(row = 0; row < LENGTH; row++){
        if(row%TILE_WIDTH == 0) printf("\n");
        for(col = 0; col < LENGTH; col++){
            if(col%TILE_WIDTH == 0) printf(" ");
            int DestIndex = row * LENGTH + col;
            printf("%d ", MT[DestIndex]);
        }
        printf("\n");
    }
    printf("\n");
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
    struct timeval cpu_start, cpu_end, cpu_gap;

    int WIDTH = 6400;
    int MatrixSize = WIDTH * WIDTH;
    int* M;
    int* MT;
    M = (int *)malloc(MatrixSize*sizeof(int));
    MT = (int *)malloc(MatrixSize*sizeof(int));

    for(int i = 0;i<MatrixSize;i++){
        M[i] = rand()%8;
        MT[i] = 0;
    }

    gettimeofday(&cpu_start, NULL);
    matrixTranspose(M,MT,WIDTH);
    gettimeofday(&cpu_end, NULL);

    getGapTime(&cpu_start, &cpu_end, &cpu_gap);
    float f_cpu_gap = timevalToFloat(&cpu_gap);

    //printResult(M,MT,WIDTH);

    printf("CPU time = %.6f\n", f_cpu_gap);

    free(M); free(MT);
    return 0;
}