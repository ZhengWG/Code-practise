# -*- coding: utf-8 -*-
'''
@author:    Wengang.Zheng
@email:     zwg0606@gmail.com
@fielName:  最长递增子序列.py
@Time:      2021-05-23-17:42:52
@Des:       最长递增子序列（LIS）：经典动态规划题目，时间复杂度O（n^2）
'''


def lengthOfLIS(nums) -> int:
    """
    @brief      求最长递增子序列
    @details    https://leetcode-cn.com/problems/longest-increasing-subsequence/
    @param      nums: 输入无序数组 
    @return     res: 最长递增子序列长度
    """
    # 初始化dp
    dp = [1] * len(nums)

    # 更新dp
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    print(dp)
    # 基于dp得到最长子序列长度
    res = max(dp)
    return res


if __name__ == '__main__':
    input1 = [0, 1, 0, 3, 2, 3]
    res1 = lengthOfLIS(input1)
    print(res1)
