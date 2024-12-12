/*
把只包含质因子2、3和5的数称作丑数（Ugly Number）。
例如6、8都是丑数，但14不是，因为它包含质因子7。 
习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
*/
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> GetUglyNumber(int index) {
        if (index <= 0)
            return vector<int>({});
        int t2 = 0, t3 = 0, t5 = 0;
        vector<int> nums(index);
        nums[0] = 1;
        for (int i=1; i<index; i++) {
            nums[i] = min(nums[t2] * 2, min(nums[t3] * 3, nums[t5] * 5));
            if (nums[i] == nums[t2] * 2)
                t2++;
            if (nums[i] == nums[t3] * 3)
                t3++;
            if (nums[i] == nums[t5] * 5)
                t5++;
        }
        return nums;
    }
};

int main() {
    Solution s = Solution();
    vector<int> nums = s.GetUglyNumber(10);
    for (auto it=nums.begin(); it!=nums.end(); it++){
        if (it != nums.end() - 1)
            printf("%d->", *it);
        else
            printf("%d", *it);
    }

}