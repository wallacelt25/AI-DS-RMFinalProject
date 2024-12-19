#include <stdio.h>
#include "mylibs/mylib.h"

int main(void){

    int n = 10;

    printf("%d! = %d = %d\n",n, fibonaccirecursive(n), 
    fibonacciiterative(n));

    for(int i = 1; i <= n; i++){
        if(fibonacciiterative(i)!= fibonaccirecursive
        (i)){
            printf("[ERROR]\n");
        }
        
        else{
            printf("F(%d) = %d = %d\n",i,
            fibonacciiterative(i), fibonaccirecursive
            (i));
        }
    }

    return 0;

}