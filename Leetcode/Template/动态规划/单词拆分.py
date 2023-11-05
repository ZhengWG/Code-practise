# -*- coding: utf-8 -*-
'''
@author:    Wengang.Zheng
@email:     zwg0606@gmail.com
@fielName:  单词拆分.py
@Time:      2023-11-05-18:04:44
@Des:       单词拆分
'''

from typing import List, Dict

class Solution1:
    """
    @brief      单词拆分一

    @details    https://leetcode.cn/problems/word-break/
    """
    def __init__(self):
        self.worddict = {}  # 字典
        self.mem = []  # 备忘录

    def word_break1(self, worddict: List[str], s: str):
        """
        @brief      单词拆分一

        @details    https://leetcode.cn/problems/word-break/
        """

        self.worddict = dict(worddict)
        # 备忘录初始化
        self.mem = [-1 for _ in range(len(s))]
        result = self.dp(s, 0)
        return result

    def word_break2(self, worddict: List[str], s: str):
        """
        @brief      单词拆分二

        @details    https://leetcode.cn/problems/word-break-ii/
        """
        self.worddict = dict(worddict)
        # 备忘录初始化
        self.mem = [-1 for _ in range(len(s))]
        result = self.dp2(s, 0)
        return result

    def dp2(self, target, i):
        res = []
        length = len(target)
        if i == length:
            res.append("")
            return res

        if self.mem[i] != -1:
            return self.mem[i]

        for length in range(1, length + 1 - i):
            word_prefix = target[i:i+length-1]
            if word_prefix in self.worddict:
                sub_res = dp(target, i+length, res)
                # 需要遍历得到所有结果
                for sub in sub_res:
                    if sub == "":
                        res.append(prefix)
                    else:
                        res.append(prefix + " " + sub)

        self.mem[i] = res
        return res

    def dp(self, target, i):
        length = len(target)
        if i == length:
            return True

        if self.mem[i] != -1:
            return self.mem[i]

        for length in range(1, length + 1 - i):
            word_prefix = target[i:i+length-1]
            if word_prefix in self.worddict:
                res = dp(target, i+length)
                if res:
                    self.mem[i] = True
                    return True
        self.mem[i] = False
        return False
