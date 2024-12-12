/*
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
*/
#include <iostream>
using namespace std;

class Solution {
public:
    int NumberOf1_1(int n){
        // 基础版本：没有考虑负数问题
        int count = 0;
        while (n) {
            if (n & 1)
                count++;
            n = n >> 1;
        }
        return count;
    }

    int NumberOf1_2(int n){
        // 优化版本：考虑负数问题,循环次数始终为int位数
        int count = 0;
        unsigned int flag = 1;
        while(flag){
            if(n & flag)
                count++;
            flag = flag << 1;
        }
        return count;
    }

    int NumberOf1_3(int n){
        // 最优版本：循环次数为1的个数
        int count = 0;
        while(n){
            n = n & (n-1); // 多少个1意味着能做多少次运算   
            count++;
        }
        return count;
    }
};


int main(){
    Solution s;
    int number = 9;
    int res1 = s.NumberOf1_1(number);
    cout << "res1 for " << number << " is " << res1 << endl;
    int res2 = s.NumberOf1_2(number);
    cout << "res2 for " << number << " is " << res2 << endl;
    int res3 = s.NumberOf1_3(number);
    cout << "res3 for " << number << " is " << res3 << endl;
}