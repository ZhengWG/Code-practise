/*
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
*/
#include <iostream>
using namespace std;

class Solution {
public:
    double Power(double base, int exponent) {
        // 递归实现：
        // 1. 奇数：result *= base
        // 2. 偶数：result *= result
        // 3. -1: 1.000/base
        // 4. 1: base
        // 5: 0: 1

        if(!exponent)
            return 1;
        if(exponent == 1)
            return base;
        if(exponent == -1)
            return 1.000/base;

        // 移位表示除以2，若为奇数，需要乘以base
        double result = Power(base, exponent>>1);
        result *= result;

        // 奇数
        if(exponent & 0x01 == 1)
            result *= base;

        return result;
    }
};

int main() {
    Solution s;
    cout << s.Power(2, 3) << endl;
    return 0;
}