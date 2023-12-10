# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 二分.py
@Time:   2020-12-13-01:02:11
@Des:
二分法的一些注意事项:
+ 对于含有重复数字的情况是返回左边界还是右边界
+ 计算mid的时候可能溢出:mid=(left+right)/2
+ left,right以及等号问题
二分法的技巧点:
+ 尽量不要用else,全部用elif
+ 采用left + (right - left) / 2
+ 采用固定的模板设计
'''


def binary_search(nums, target):
    """
    @brief      最简单的二分查找
    """
    left = 0
    right = len(nums) - 1
    # left, righto作等号
    while (left <= right):
        mid = left + (right - left) // 2
        # left/right直接更新
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        elif nums[mid] == target:
            # 直接返回,不考虑边界
            return mid


def left_bound_search(nums, target):
    """
    @brief      查找左边界
    """
    left = 0
    right = len(nums) - 1
    while (left <= right):
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        elif nums[mid] == target:
            # 不返回，锁定左边界
            right = mid - 1

    #　最后需要检查left越界情况
    if left >= len(nums) or nums[left] != target:
        return -1
    return left


def right_bound_search(nums, target):
    """
    @brief      查找右边界
    """
    left = 0
    right = len(nums) - 1
    while (left <= right):
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        elif nums[mid] == target:
            # 不返回，锁定右边界
            left = mid + 1

    # 最后检查right越界情况
    if right < 0 or nums[right] != target:
        return -1
    return right
