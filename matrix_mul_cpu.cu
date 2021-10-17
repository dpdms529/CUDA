
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LENGTH 12

void MatrixMul(int* M, int* N, int* P){
    int row = 0; int col = 0;
    for(row = 0; row < LENGTH; row++){
        for(col = 0; col < LENGTH; col++){
            int Destindex = row * LENGTH + col;
            for(int index = 0; index<LENGTH; index++){
                P[Destindex] += M[row * LENGTH + index] + N[col + index * LENGTH];
            }
            
        }
    }
}

void printResult(int* M, int* N, int* P){
    int row = 0; int col = 0;
    for(row = 0; row<LENGTH; row++){
        for(col = 0; col<LENGTH; col++){
            int Destindex = row * LENGTH + col;
            for(int index = 0;index<LENGTH;index++){
                printf("(%d = A[%d][%d], %d = B[%d][%d]) \n",
                   M[row * LENGTH + index], row, index, N[col + index * LENGTH], index, col);
            }
            printf("%d, C[%d][%d] = A[%d][.] dot B[.][%d] \n\n",
                   P[Destindex], row, col, row, col);
            
        }
    }
}

int main(){
    srand(time(NULL));

    const int MatrixWidth = LENGTH; const int MatrixHeight = LENGTH;
    const int MatrixSize = MatrixWidth * MatrixHeight;
    const int BufferSize = MatrixSize * sizeof(int);

    int* M; int* N; int* P_C;

    M = (int*)malloc(BufferSize);
    N = (int*)malloc(BufferSize);
    P_C = (int*)malloc(BufferSize);

    for(int i = 0;i< MatrixSize;i++){
        M[i] = rand()%4; N[i] = rand()%8; P_C[i] = 0;
    }
    MatrixMul(M,N,P_C);
    printResult(M,N,P_C);

    free(M); free(N); free(P_C);

    return 0;
}