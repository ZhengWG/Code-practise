/*
一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。
请写程序找出这两个只出现一次的数字。
*/
#include <iostream>
using namespace std;

class Solution{
public:
    void FindNumsAppearOnce(vector<int> data, int* num1, int *num2) {
        // 核心思路：
        // 1. 先求出所有数字的异或结果：为两个只出现一次的数字的异或结果
        // 2. 找到异或结果中为1的位，说明两个数字在该位上异号，可以通过该位将数组分为两组
        // 3. 每组中分别求异或结果，即为两个只出现一次的数字

        // 全部求异或
        int sum = 0;
        for(auto &num : data){
            sum = sum ^ num;
        }

        // 找到异或结果中为1的位
        int index = 0;
        while(1){
            if(sum & 0x01) break;
            sum = sum >> 1;
            index++;
        }

        // 分组求异或
        int n1 = 0, n2 = 0;
        int mask = 1 << index;
        for(auto &num : data){
            if(num & mask)
                n1 = n1 ^ num;
            else
                n2 = n2 ^ num;
        }
        *num1 = n1;
        *num2 = n2;
    }
};

int main(){
    Solution s;
    vector<int> data = {1, 1, 3, 4, 4, 6, 6, 8, 9, 9};
    int num1, num2;
    s.FindNumsAppearOnce(data, &num1, &num2);
    cout << num1 << ", " << num2 << endl;
    return 0;
}