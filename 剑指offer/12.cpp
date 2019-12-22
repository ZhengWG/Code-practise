/*
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
*/
class Solution {
public:
    double Power(double base, int exponent) {
        if(!exponent)
            return 1;
        if(exponent == 1)
            return base;
        if(exponent == -1)
            return 1.000/base;
		//移位表示除以2，若为奇数，需要乘以base
        double result = Power(base,exponent>>1);
        result *= result;
        if(exponent & 0x01 == 1)
            result *= base;
        return result;
    }
};
