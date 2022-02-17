
#include <stdio.h>
#include <sys/time.h>

int main(){
    time_t timer;
    time(&timer);
    printf("1970년 1월 1일 0시 이후로 %ld 초가 지났습니다. \n",timer);

    struct timeval utimer;
    gettimeofday(&utimer, NULL);
    printf("1970년 1월 1일 0시 이후로 %ld 초 및 %ld 마이크초가 지났습니다.", utimer.tv_sec, utimer.tv_usec);
}