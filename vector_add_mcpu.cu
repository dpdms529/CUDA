
#include <stdio.h>

#define N 20
#define K 4

void add(int myid, int *a, int *b, int *c){
    int tid = myid;
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += K;
    }
}

int main(void){
    int a[N], b[N], c[N];

    for(int i = 0;i<N;i++){
        a[i] = i;
        b[i] = i*i;
    }

    add(0,a,b,c); add(1,a,b,c); add(2,a,b,c); add(3,a,b,c);

    for (int i = 0;i<N;i++){
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;

}