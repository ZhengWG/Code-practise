/*
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。
求该青蛙跳上一个n级的台阶总共有多少种跳法。
*/
class Solution {
public:
	//相当于比每次都比前一阶台阶多一倍的选择（可以跳到那节台阶也可以不跳）
    int jumpFloorII(int number) {
        if(!number)
          return 0;
        long long num=1;
        for(int i=1;i<number;i++)
            num*=2;
        return num;
    }
};