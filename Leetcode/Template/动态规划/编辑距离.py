# -*- coding: utf-8 -*-
'''
@author:    Wengang.Zheng
@email:     zwg0606@gmail.com
@fielName:  编辑距离.py
@Time:      2021-05-23-18:02:52
@Des:       核心步骤：
            递归解法：
            1. 递归框架：skip/插入/删除/替换
            2. 备忘录解决重叠子问题
            迭代DP table：
            1. 构建DP table：注意初始化状态
            2. 更新状态：skip/插入/删除/替换
'''


def minDistance1(s1, s2) -> int:
    """
    @brief      最小编辑距离递归解法
    @details    https://leetcode-cn.com/problems/edit-distance
    @param      s1: 输入字符1
                s2：输入字符2
    @return     res -> int: 最小编辑距离
    """
    mem = {}

    def dp(i, j):
        if (i, j) in mem:
            return mem[(i, j)]
        if i == -1:
            # 全部插入
            return j + 1
        if j == -1:
            return i + 1

        if s1[i] == s2[j]:
            return dp(i - 1, j - 1)
        else:
            return min(
                dp(i - 1, j) + 1, # 删除
                dp(i, j - 1) + 1, # 插入
                dp(i - 1, j - 1) + 1 # 替换
            )

    return dp(len(s1) - 1, len(s2) - 1)


def minDistance2(s1, s2) -> int:
    """
    @brief      最小编辑距离迭代解法
    @details    https://leetcode-cn.com/problems/edit-distance
    @param      s1: 输入字符1
                s2：输入字符2
    @return     res -> int: 最小编辑距离
    """
    # 初始化DP table
    dp = [[0] * (len(s1) + 1) for _ in range(len(s2) + 1)]
    for i in range(len(s2) + 1):
        dp[i][0] = i
    for i in range(len(s1) + 1):
        dp[0][i] = i

    for i in range(len(s2)):
        for j in range(len(s1)):
            if s1[j] == s2[i]:
                dp[i + 1][j + 1] = dp[i][j]
            else:
                dp[i + 1][j + 1] = min(
                    dp[i][j + 1] + 1,
                    dp[i + 1][j] + 1,
                    dp[i][j] + 1
                )
    return dp[-1][-1]

if __name__ == '__main__':
    s1 = 'intention'
    s2 = 'execution'
    res1 = minDistance1(s1, s2)
    res2 = minDistance2(s1, s2)
    assert res1 == 5 and res2 == 5
    s1 = 'horse'
    s2 = 'ros'
    res1 = minDistance1(s1, s2)
    res2 = minDistance2(s1, s2)
    assert res1 == 3 and res2 == 3
    print('All test passed!')
