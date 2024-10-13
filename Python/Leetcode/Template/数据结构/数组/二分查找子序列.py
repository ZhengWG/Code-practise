# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 二分查找子序列.py
@Time:   2021-01-31-23:47:46
@Des:　难点在多个子序列的匹配优化
'''


def isSubsequence(s, t):
    """
    @brief      判断s是否为t的子序列

    @details    https://leetcode-cn.com/problems/is-subsequence/
　　　　　　　　时间复杂度为O(n),n=len(t)

    @param      s(str): 短序列
    　　　　　　t(str): 长序列

    @return     (bool): 是否为子序列
    """
    # 若s为空，则为True
    if not len(s):
        return True
    i = j = 0
    for j in range(len(t)):
        if t[j] == s[i]:
            i += 1
        if i >= len(s):
            return True
    return False


def left_bound(arr, tar):
    """
    @brief      二分查找左边界
    """
    lo = 0
    hi = len(arr)

    while(lo < hi):
        mid = (lo + hi) // 2
        if tar > arr[mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def isSubsequence_multi(s_l, t):
    """
    @brief      判断s是否为t的子序列

    @details    https://leetcode-cn.com/problems/is-subsequence/
　　　　　　　　单个s时间复杂度优化为O(mlog(n)),n=len(t),m=len(s)

    @param      s(list): 短序列list
    　　　　　　t(str): 长序列

    @return     res(list): 是否为子序列list
    """
    # 需要先对长序列作预处理，存储各个字符出现的位置list
    index = {}
    for i in range(len(t)):
        c = t[i]
        if c not in index:
            index[c] = [i]
        else:
            index[c].append(i)

    # 依次判断是否满足子序列
    res = []
    for s in s_l:
        # 初始化t中匹配的序号
        pos_t = 0
        is_math = True
        for j in range(len(s)):
            c = s[j]
            if c not in index:
                # 无法匹配
                is_math = False
                break
            # 查找左边界
            pos = left_bound(index[c], pos_t)
            if pos == len(index[c]):
                # 没找到
                is_math = False
                break
            pos_t = index[c][pos] + 1
        res.append(is_math)

    return res


if __name__ == '__main__':
    s1 = 'abc'
    t1 = 'ahbgdc'
    res1 = isSubsequence(s1, t1)
    res1_l = isSubsequence_multi([s1], t1)
    assert res1 == True, 'test1 failed!'
    assert res1 == res1_l[0], 'test1 multi failed!'

    s2 = 'axc'
    t2 = 'ahbgdc'
    res2 = isSubsequence(s2, t2)
    res2_l = isSubsequence_multi([s2], t2)
    assert res2 == False, 'test2 failed!'
    assert res2 == res2_l[0], 'test2 multi failed!'
    print("All test passed!")
