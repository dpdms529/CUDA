
#include <stdio.h>
#include <time.h>

int main(){
    time_t timer;
    time(&timer);
    char* curTime = ctime(&timer);

    printf("1970년 1월 0시 이후로 %ld 초가 지났습니다. \n", timer);
    printf("현재 시간은 \n%s입니다. \n", curTime);
}