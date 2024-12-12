/*
斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39
*/
#include <iostream>

class Solution {
public:
    int Fibonacci(int n) {
        if (n==0)
            return 0;
        int num_one = 0;
        int num_two = 1;
        for (int i=2; i<=n; i++){
            int tmp = num_two;
            num_two = num_two + num_one;
            num_one = tmp;
        }
        return num_two;
    }
};

int main(){
    Solution sol;
    int res = sol.Fibonacci(10);
    std::cout << "res:" << res << std::endl;
}