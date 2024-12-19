#include <stdio.h>

//Recursive
int fibonaccirecursive(int n){
    if (n == 0){
        return 0;
    }
    else if (n == 1){ 
        return 1;
    }
    else{
        return fibonaccirecursive(n - 1) + fibonaccirecursive(n - 2);
    }
}

//Iterative
int fibonacciiterative(int n)
{
    int i, one = 0, two = 1, three;
    if (n == 0)
        return one;
    for (i = 2; i <= n; i++)
    {
        three = one + two;
        one = two;
        two = three;
    }
    return two;
}
