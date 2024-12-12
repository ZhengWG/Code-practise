/*
一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。
请写程序找出这两个只出现一次的数字。
*/
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> findNumsAppearOnce(vector<int>& nums) {
        // Step 1: Get XOR of all numbers - this will give XOR of two unique numbers
        int xorResult = 0;
        for (int num : nums) {
            xorResult ^= num;
        }
        
        // Step 2: Find rightmost set bit in xorResult
        int rightmostSetBit = 1;
        while ((xorResult & rightmostSetBit) == 0) {
            rightmostSetBit <<= 1;
        }
        
        // Step 3: Divide numbers into two groups based on the rightmost set bit
        // and XOR them separately
        int x = 0, y = 0;
        for (int num : nums) {
            if (num & rightmostSetBit) {
                x ^= num;
            } else {
                y ^= num;
            }
        }
        
        return {min(x, y), max(x, y)};
    }
};
