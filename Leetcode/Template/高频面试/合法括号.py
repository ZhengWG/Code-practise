# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 合法括号.py
@Time:   2021-01-17-16:57:00
@Des:
判断字符串的括号是否合法：
常用数据结构：栈
'''


class Stack(object):
    def __init__(self):
        self.stack = []

    def push(self, data):
        """
        @brief      进栈操作
        """
        self.stack.append(data)

    def pop(self):
        """
        @brief      出栈操作
        """
        return self.stack.pop()

    def gettop(self):
        """
        @brief      取栈顶
        """
        return self.stack[-1]


def isValid(input_s):
    """
    @brief      判断字符串的括号是否合法
    """
    check_dict = {
        ')': '(',
        '}': '{',
        ']': '['
    }
    l_list = ['(', '{', '[']
    r_list = [')', '}', ']']
    check_stack = Stack()

    for s in input_s:
        if s in l_list:
            check_stack.push(s)
        elif s in r_list:
            if len(check_stack.stack) and check_stack.gettop() == check_dict[s]:
                # 一对括号匹配
                check_stack.pop()
            else:
                return False

    # 判断左括号是否全部match
    return len(check_stack.stack) == 0
