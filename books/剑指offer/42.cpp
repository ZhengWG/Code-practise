/*
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
输出描述:
对应每个测试案例，输出两个数，小的先输出。
*/
class Solution {
public:
	//前后追赶即可
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        vector<int> out;
        if(array.size() < 2)
            return out;
        int small = 0;
        int big =  array.size()-1;
        while(small < big){
            int tmp = array[small]+array[big];
            if(tmp==sum){
                out.push_back(array[small]);
                out.push_back(array[big]);
                break;
            }
            else if(tmp < sum)
                small++;
            else
                big--;
        }
        return out;
    }
};