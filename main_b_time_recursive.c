#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include "mylibs/mylib.h"

int main(void) {
        srand(time(0));

        int w;

        const int n = 40;

        float startTime = (float)clock()/CLOCKS_PER_SEC;
        for (int i= 0; i <= n; i++){
                w = fibonacciiterative(i);
        }
        float endTime  = (float)clock()/CLOCKS_PER_SEC;
        printf("Time elapsed: %f s\n",endTime - startTime);
}