/*
0,1,…,n-1这n个数字排成一个圆圈，从数字0开始每次从这个圆圈里删除第m个数字。
求出这个圆圈里剩下的最后一个数字。
*/
#include <iostream>
#include <vector>
using namespace std;

class Solution{
public:
    int LastRemaining_Solution(int n, int m) {
        // 核心思路：vector构造环形链表来模拟
        if(n < 1 || m < 1)
            return -1;
        vector<int> nums(n);
        for(int i = 0; i < n; i++)
            nums[i] = i;
        int index = 0;
        while(nums.size() > 1){
            index = (index + m - 1) % nums.size();
            nums.erase(nums.begin() + index);
        }
        return nums[0];
    }
};

int main(){
    Solution s;
    cout << s.LastRemaining_Solution(5, 3) << endl;
    return 0;
}