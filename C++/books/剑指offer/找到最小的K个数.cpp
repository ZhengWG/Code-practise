/*
输入n个整数，找出其中最小的K个数。
例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
*/
#include <iostream>
#include <vector>
#include <set>

using namespace std;


class Solution {
public:
    vector<int> GenLeastNumber_Solution(vector<int> input, int k) {
        vector<int> out;
        if (!k || k>input.size())
            return input;

        set<int> nums;  // set自排序
        for (int i=0; i<input.size(); i++) {
            if (nums.size() < k)
                nums.insert(input[i]);
            else {
                auto it = nums.end();
                it--;
                if (*it > input[i]) {
                    nums.erase(*it);
                    nums.insert(input[i]);
                }
            }
        }
        for (auto it=nums.begin(); it!=nums.end(); it++) 
            out.push_back(*it);
        return out;
    }
};