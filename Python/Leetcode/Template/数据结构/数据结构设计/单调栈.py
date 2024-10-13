# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 单调栈.py
@Time:   2020-12-21-00:19:03
@Des:
单调栈用途较少：仅解决单一问题: Next Greater Element
'''


def nextGreatElement(nums: list):
    """
    @brief      下一个更大元素：找到对应index的下一个更大元素，否则返回-1

    @details    https://leetcode-cn.com/problems/next-greater-element-i/

    @param      nums: 输入元素list

    @return     res: 更大元素list
    """
    res = [-1] * len(nums)
    # 栈用list模拟
    stack_s = []
    # 倒着入栈
    for i in range(len(nums)):
        idx = len(nums) - 1 - i
        # 判断数目大小：单调栈中仅保留比当前nums[idx]更大的元素:原因是更小的元素会被压缩不会用到
        while (len(stack_s) and stack_s[len(stack_s) - 1] <= nums[idx]):
            stack_s.pop()

        # 保存更大元素
        res[idx] = stack_s[len(stack_s) - 1] if len(stack_s) else -1
        stack_s.append(nums[idx])

    return res


# test for nextGreatElement
nums = [2, 1, 2, 4, 3]
res = nextGreatElement(nums)
print(res)
