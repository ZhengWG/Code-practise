# -*- coding: utf-8 -*-
'''
@author:    Wengang.Zheng
@email:     zwg0606@gmail.com
@fielName:  最长公共子序列.py
@Time:      2021-05-23-18:04:44
@Des:       动态规划经典：最长公共子序列（LCS）：二维动态规划
'''


# 递归解法
def longestCommonSubsequence1(s1, s2) -> int:
    """
    @brief      最长公共子序列递归解法
    @details    https://leetcode-cn.com/problems/longest-common-subsequence
    @param      s1: 字符1
                s2：字符2
    @return     res：最长公共子序列长度
    """
    def dp(i, j):
        # 递归终止条件
        if i == -1 or j == -1:
            return 0

        if s1[i] == s2[j]:
            return dp(i - 1, j - 1) + 1
        else:
            return max(dp(i - 1, j), dp(i, j - 1))

    return dp(len(s1) - 1, len(s2) - 1)


# 迭代解法
def longestCommonSubsequence2(s1, s2) -> int:
    """
    @brief      最长公共子序列迭代解法
    @details    迭代更新dp table：https://leetcode-cn.com/problems/longest-common-subsequence
    @param      s1: 字符1
                s2：字符2
    @return     res：最长公共子序列长度
    """
    # 构建二维DP table
    dp = [ [0] * (len(s1) + 1) for _ in range(len(s2) + 1)]

    for i in range(len(s2)):
        for j in range(len(s1)):
            if s2[i] == s1[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[-1][-1]


if __name__ == '__main__':
    s1 = 'abcde'
    s2 = 'ace'
    res1 = longestCommonSubsequence1(s1, s2)
    res2 = longestCommonSubsequence2(s1, s2)
    assert res1 == 3 and res2 == 3
    print("All test passed!")
