# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 编辑距离.py
@Time:   2021-01-10-15:46:23
@Des:
最小编辑距离:通常思路为采用两个指针i,j分别指向两个字符串的最后,然后一步步往前走,缩小问题的规模
'''


def minDistance(s1, s2):
    """
    @brief      得到两个字符串的最小编辑距离,可用的编辑操作:
                + 插入一个字符
                + 删除一个字符
                + 替换一个字符

    @details    https://leetcode-cn.com/problems/edit-distance/

    @param      s1(str): 字符1
                s2(str): 字符2

    @return     min_dis(int): 最小编辑距离
    """
    m = len(s1)
    n = len(s2)

    # 构建dp table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # base case
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 自底向上求解dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                # 字符相同, 直接跳过
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 字符不同,需要操作:替换/插入/删除
                dp[i][j] = min(
                    dp[i - 1][j] + 1, # 插入
                    dp[i][j - 1] + 1, # 删除
                    dp[i - 1][j - 1] + 1, # 替换
                )

    return dp[m][n]


def minDistance_v2(s1, s2):
    """
    @brief      得到两个字符串的最小编辑距离,可用的编辑操作:
                + 插入一个字符
                + 删除一个字符
                + 替换一个字符

    @details    https://leetcode-cn.com/problems/edit-distance/

    @param      s1(str): 字符1
                s2(str): 字符2

    @return     dp(list): 最小编辑距离dp
    """
    m = len(s1)
    n = len(s2)

    # 构建dp table
    class Node:
        def __init__(self, value, choice):
            self.value = value
            # 0: 插入,1:删除,2:替换,3:跳过
            self.choice = choice
    dp = [[Node(value=0, choice=-1)] * (n + 1) for _ in range(m + 1)]
    # base case
    for i in range(m + 1):
        dp[i][0] = Node(value=i, choice=0)
    for j in range(n + 1):
        dp[0][j] = Node(value=j, choice=0)

    def n_min(node1, node2, node3):
        import numpy as np
        idx = np.argmin([node1.value, node2.value, node3.value])
        return idx

    # 自底向上求解dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                # 字符相同, 直接跳过
                value = dp[i - 1][j - 1].value
                dp[i][j] = Node(value=value, choice=3)
            else:
                # 字符不同,需要操作:替换/插入/删除
                idx = n_min(
                    dp[i - 1][j], # 插入
                    dp[i][j - 1], # 删除
                    dp[i - 1][j - 1], # 替换
                )
                node = [dp[i -1][j], dp[i][j - 1], dp[i - 1][j - 1]][idx]
                dp[i][j] = Node(value=node.value + 1, choice=idx)

    return dp


if __name__ == "__main__":
    s1 = 'intention'
    s2 = 'execution'
    re1 = minDistance(s1, s2)
    s1 = 'horse'
    s2 = 'ros'
    re2 = minDistance(s1, s2)
    assert re1 == 5 and re2 == 3
    print("All test passed!")
    s1 = 'horse'
    s2 = 'ros'
    re3 = minDistance_v2(s1, s2)
