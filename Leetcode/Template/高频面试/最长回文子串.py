# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 最长回文子串.py
@Time:   2021-01-17-15:45:41
@Des:
回文子串:左右对称,主要存在两种情况:奇数/偶数,如:
cac,cc
'''


def palindrome(s, left, right):
    """
    @brief      判断以l/r位置为中心的最长回文子串

    @param      s: 输入字串
                l: 左位置
                r: 右位置(l+1/l)

    @return     re_s: 最长回文子串
    """
    # 判断回文:索引不能越界且左右对应字符相同
    while (left >= 0 and right < len(s) and s[left] == s[right]):
        left -= 1
        right += 1

    return s[left + 1: right]


def longestPalindrome(s):
    """
    @brief      找到字符的最大回文子串

    @details    思路:寻找以每个单个字符/相邻相等字符为中心的最大回文子串,

    @param      s: 输入字符

    @return     re_s: 最大回文字符
    """
    re_s = s[0]
    for idx in range(len(s) - 1):
        s1 = palindrome(s, idx, idx)
        s2 = palindrome(s, idx, idx + 1)
        if len(s1) > len(re_s):
            re_s = s1
        if len(s2) > len(re_s):
            re_s = s2

    return re_s


if __name__ == '__main__':
    input_s1 = 'cbbd'
    input_s2 = 'babad'
    input_s3 = 'ac'

    re1 = longestPalindrome(input_s1)
    re2 = longestPalindrome(input_s2)
    re3 = longestPalindrome(input_s3)
    assert re1 == 'bb' and re2 == 'bab' and re3 == 'a'
    print("All test passed!")
