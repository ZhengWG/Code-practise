# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 接雨水.py
@Time:   2021-01-17-14:53:55
@Des:
核心思路:i位置的蓄水量由左右两边的最高柱子高度决定
water[i] = min(
# 左边最高的柱子
max(height[0..i]),
# 右边最高的柱子
max(height[i..end])
) - height[i]
'''


def violent_trap(heights):
    """
    @brief      接雨水暴力解法: O(n^2),存在重复计算

    @details    https://leetcode-cn.com/problems/trapping-rain-water/

    @param      heights: list for height

    @return     res: num for trapping water(int)
    """
    res = 0
    for idx in range(len(heights)):
        # 找到左边最高的柱子
        max_l = heights[idx]
        for l_idx in range(idx):
            max_l = max(max_l, heights[l_idx])

        max_r = heights[idx]
        for r_idx in range(idx + 1, len(heights)):
            max_r = max(max_r, heights[r_idx])

        res += min(max_l, max_r) - heights[idx]

    return res


def dp_trap(heights):
    """
    @brief      接雨水备忘录解法: 时间复杂度:O(n),空间复杂度可优化

    @details    https://leetcode-cn.com/problems/trapping-rain-water/

    @param      heights: list for height

    @return     res: num for trapping water(int)
    """
    from copy import deepcopy
    res = 0
    table_l = deepcopy(heights)
    table_r = deepcopy(heights)

    # 计算table_l: O(n)
    for idx in range(1, len(heights)):
        table_l[idx] = max(heights[idx], table_l[idx - 1])

    # 计算table_r: O(n)
    for idx in range(len(heights) - 1):
        r_idx = len(heights) - idx - 1
        table_r[r_idx] = max(heights[r_idx], table_r[r_idx + 1])

    # 根据table计算蓄水量
    for idx in range(len(heights)):
        res += min(table_l[idx], table_r[idx]) - heights[idx]

    return res


def twopoint_trap(heights):
    """
    @brief      接雨水双指针解法:时间/空间复杂度:O(n)

    @details    https://leetcode-cn.com/problems/trapping-rain-water/

    @param      heights: list for height

    @return     res: num for trapping water(int)
    """
    left = 0
    right = len(heights) - 1
    l_max = heights[0]
    r_max = heights[-1]
    res = 0

    # 指针从两端向中间聚拢
    # 通过判断左右的height值判断雨水量
    while (left < right):
        # 更新左右高度极值
        l_max = max(l_max, heights[left])
        r_max = max(r_max, heights[right])

        if l_max < r_max:
            # 左边高度更小:雨水量由左边最高柱子决定
            res += l_max - heights[left]
            left += 1
        else:
            # 右边高度更小:雨水量由右边最高柱子决定
            res += r_max - heights[right]
            right -= 1
    return res


if __name__ == '__main__':
    heights = [4, 2, 0, 3, 2, 5]
    res1 = violent_trap(heights)
    res2 = dp_trap(heights)
    res3 = twopoint_trap(heights)
    assert res1 == 9 and res2==9 and res3==9
    print("All test passed!")
