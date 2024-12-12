/*
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
输出描述:
对应每个测试案例，输出两个数，小的先输出。
*/
#include <iostream>
#include <vector>
using namespace std;

class Solution{
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        // 核心思路：前后指针
        // 如果和小于sum，则增大前指针，如果和大于sum，则减小后指针
        // 如果和等于sum，则记录结果，并减小后指针
        // 最后返回最后一个结果
        vector<pair<int, int>> out;
        int small = 0, big = array.size() - 1;
        while (small < big){
            int tmp = array[small] + array[big];
            if(tmp == sum){
                out.push_back(make_pair(array[small], array[big]));
                big--;
            }
            else if(tmp < sum)
                small++;
            else
                big--;
        }
        // for debug
        for(auto &o : out){
            cout << o.first << ", " << o.second << endl;
        }
        auto it = out.begin();
        return vector<int>{it->first, it->second};
    }
};

int main(){
    Solution s;
    vector<int> array = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int sum = 10;
    vector<int> out = s.FindNumbersWithSum(array, sum);
    cout << out[0] << ", " << out[1] << endl;
    return 0;
}