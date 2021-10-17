
#include <stdio.h>

#define LENGTH 12

void MatrixAdd(int* M, int* N, int* P){
    int row = 0; int col = 0;
    for(row = 0; row < LENGTH; row++){
        for(col = 0; col < LENGTH; col++){
            int Destindex = row * LENGTH + col;
            P[Destindex] = M[Destindex] + N[Destindex];
        }
    }
}

void printResult(int* M, int* N, int* P){
    int row = 0; int col = 0;
    for(row = 0; row<LENGTH; row++){
        for(col = 0; col<LENGTH; col++){
            int Destindex = row * LENGTH + col;
            printf("%d (= C[%d][%d]) = %d (= A[%d][%d]) + %d (= B[%d][%d]) \n",
                   P[Destindex], row, col, M[Destindex], row, col, N[Destindex], row, col);
        }
    }
}

int main(){
    const int MatrixWidth = LENGTH; const int MatrixHeight = LENGTH;
    const int MatrixSize = MatrixWidth * MatrixHeight;
    const int BufferSize = MatrixSize * sizeof(int);

    int* M; int* N; int* P_C;
    M = (int*)malloc(BufferSize);
    N = (int*)malloc(BufferSize);
    P_C = (int*)malloc(BufferSize);

    for(int i = 0;i< MatrixSize;i++){
        M[i] = i; N[i] = i; P_C[i] = 0;
    }
    MatrixAdd(M,N,P_C);
    printResult(M,N,P_C);

    free(M); free(N); free(P_C);

    return 0;
}