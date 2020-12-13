# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 滑动窗口.py
@Time:   2020-12-13-13:42:18
@Des:
滑动窗口模板:
# 初始化
left = right = 0
while (right < len(a_s)):
    windows[a_s[right]] += 1
    right += 1

    while (valid):
        windows[a_s[right]] -= 1
        left += 1
'''


def solve_minsub_1(a_s, t_s):
    """
    @brief      最小覆盖子串问题:暴力解法

    @details    https://leetcode-cn.com/problems/minimum-window-substring/solution/

    @param      a_s:总字符，t_s:目标子串

    @return     如果存在总字符的连续子串包含目标子串所有字母则返回最短子串，否则返回''
    """
    #　暴力解法:时间复杂度:O(n^2*m)
    len_s = len(a_s)
    len_t = len(t_s)

    def transfer_dic(s):
        # 记录target字符出现的次数:可重复
        dic_t = {}
        for _ in s:
            if _ not in dic_t:
                dic_t[_] = 0
            dic_t[_] += 1
        return dic_t

    dic_t = transfer_dic(t_s)
    out_s = ''
    for i in range(len_s):
        for j in range(i + len_t, len_s):
            select_s = a_s[i:j + 1]
            dic_s = transfer_dic(select_s)
            # 判断是否覆盖
            flag = True
            for _ in dic_t:
                if _ not in dic_s:
                    flag = False
                    break
                if dic_s[_] < dic_t[_]:
                    flag = False
                    break
            if flag:
                if out_s != '':
                    if len(out_s) >= len(select_s):
                        out_s = select_s
                else:
                    out_s = select_s

    return out_s


def solve_minsub_2(a_s, t_s):
    """
    @brief      最小覆盖子串问题:滑动窗口
                思路简介:左右指针,先移动右指针到满足要求的位置,再更新左指针,\
                到左指针不满足要求再更新右指针
    　　　　　　时间复杂度:O(n+m)
    """
    # 初始化左右边界
    left = right = 0
    # 初始化匹配字符数目
    match = 0
    # 初始化字符左边界和字符长度
    start = 0
    minlen = float('INF')
    # 初始化窗口和目标字符
    windows = {}
    target_dic = {}
    for s in t_s:
        if s not in target_dic:
            target_dic[s] = 0
        target_dic[s] += 1
    target_num = len(target_dic)

    while (right < len(a_s)):
        c1 = a_s[right]
        if c1 in target_dic:
            # 存在目标字符,match次数累计
            if c1 not in windows:
                windows[c1] = 0
            windows[c1] += 1
            # 判断是否存在字符满足匹配次数
            if windows[c1] == target_dic[c1]:
                match += 1
        right += 1

        # 判断window是否覆盖最小子串
        while (match == target_num):
            # 更新最小字符
            if (right - left < minlen):
                start = left
                minlen = right - left
            # 更新windows匹配情况
            c2 = a_s[left]
            if c2 in target_dic:
                if windows[c2] > 0:
                    windows[c2] -= 1
                if windows[c2] < target_dic[c2]:
                    match -= 1
            left += 1

    if minlen == float('INF'):
        return ''
    else:
        return a_s[start : start + minlen]


def solve_findallanagrams(a_s, t_s):
    """
    @brief      找到字符串中所有字母异位词,字母异位词:字母相同，但排列不同的子串

    @details    https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/

    @param      a_s:总字符，t_s:目标子串

    @return　　 返回所有字母异位词子串的起始坐标，模板与上文的最小覆盖子串一致
    """
    # 初始化变量
    left = right = start = match = 0
    windows = {}
    target_s = {}
    out_starts = []
    for c in t_s:
        if c not in target_s:
            target_s[c] = 0
        target_s[c] += 1

    while right < len(a_s):
        # 更新right字符
        c1 = a_s[right]
        if c1 in target_s:
            if c1 not in windows:
                windows[c1] = 0
            windows[c1] += 1
            if windows[c1] == target_s[c1]:
                match += 1
        right += 1

        # 更新left
        while match == len(target_s):
            # 判断长度是否一致
            if right - left == len(target_s):
                out_starts.append(left)
            c2 = a_s[left]
            if c2 in target_s:
                windows[c2] -= 1
                if windows[c2] < target_s[c2]:
                    match -= 1
            left += 1

    return out_starts


def solve_findsubstr_norepeat(a_s):
    """
    @brief      找到字符串中不含有重复字符的最长子串长度

    @details    https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/

    @param      a_s:总字符

    @return　　 返回最长不重复子串长度
    """
    # 初始化
    left = right = max_len = 0
    window = {}

    while right < len(a_s):
        # 更新right
        c1 = a_s[right]
        if c1 not in window:
            window[c1] = 0
        window[c1] += 1
        right += 1
        # 判断是否重复,重复则更新left
        while window[c1] > 1:
            c2 = a_s[left]
            window[c2] -= 1
            left += 1
        # 更新长度
        if right - left > max_len:
            max_len = right - left

    return max_len


if __name__ == '__main__':
    # test1
    a_s = "ADOBECODEBANC"
    t_s = "ABC"
    out = solve_minsub_1(a_s, t_s)
    assert out == 'BANC', '{} != BANC'.format(out)
    # test2
    out = solve_minsub_2(a_s, t_s)
    assert out == 'BANC', '{} != BANC'.format(out)
    # test3
    a_s = 'cbaebabacd'
    t_s = 'abc'
    out = solve_findallanagrams(a_s, t_s)
    assert out == [0, 6], '{} != [0, 6]'.format(out)
    # test4
    a_s = 'abcabcbb'
    out = solve_findsubstr_norepeat(a_s)
    assert out == 3, '{} != 3'.format(out)

    print("All test passed!")
