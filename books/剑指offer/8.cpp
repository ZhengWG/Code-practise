/*
一只青蛙一次可以跳上1级台阶，也可以跳上2级。
求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
*/
class Solution {
public:
	//每次跳一步或者两步，其实就是不断迭代的过程
    int jumpFloor(int number) {
        if(!number)
            return 0;
        int num_one = 0;
        int num_two = 1;
        for(int i=2;i<=number+1;i++){
            int tmp = num_two;
            num_two = num_one + num_two;
            num_one = tmp;
        }
        return num_two;
    }
};
};