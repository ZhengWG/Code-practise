# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 数据缺失或重复.py
@Time:   2021-01-17-17:22:07
@Des:
常用于数组（尤其是元素和索引成对匹配）的方法：
排序/异或/映射
'''


def missingNumber(nums):
    """
    @brief      去除消失元素

    @details    nums: 维度为n的数组
                数组包含了0-n的数字,但是缺失一个数字,需要找出该数字

    @param      nums: 输入数组

    @return     res: 缺失元素
    """
    n = len(nums)
    res = 0
    # 先与多加的索引异或
    res ^= n
    # 依次和元素以及索引进行异或操作
    for idx in range(len(nums)):
        res ^= idx ^= nums[idx]
    return res


def findErrorNums(nums):
    """
    @brief      找出数组中缺失/重复的元素

    @details    数组大小n,存储了1-N的元素
                数组缺失了一个元素同时重复了一个元素

    @param      nums: 错误数组

    @return     miss_num: 缺失的元素
                dup_num: 重复的元素
    """
    dup_num = -1
    miss_num = -1
    for idx in range(len(nums)):
        # 元素从1开始: idx和nums元素映射
        index = abs(nums[idx]) - 1
        if nums[index] < 0:
            # 已经映射过一次,说明产生了重复
            dup_num = abs(nums[index])
        else:
            nums[index] *= -1

    for idx in range(len(nums)):
        if nums[idx] > 0:
            # 将索引转化为元素值
            miss_num = idx + 1

    return miss_num, dup_num
