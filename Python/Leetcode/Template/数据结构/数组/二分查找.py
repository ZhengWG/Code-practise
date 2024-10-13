# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 二分查找.py
@Time:   2021-01-10-16:50:12
@Des:
如果能够进行连续的空间线性搜索,便可以通过二分查找进行优化
'''


def CanFinish(piles, speed, H):
    """
    @brief      给定速度,是否可以在H内完成
    """
    idx = 0
    time = 0
    from copy import deepcopy
    p = deepcopy(piles)
    while(idx < len(p)):
        time += 1
        p[idx] -= speed
        if p[idx] <= 0:
            idx += 1
    if time > H:
        return False
    return True


def minEatingSpeed_v1(piles, H):
    """
    @brief      香蕉问题

    @details    https://leetcode-cn.com/problems/koko-eating-bananas/

    @param      piles: 香蕉堆数
                H: 时间

    @return     K: 最小速度
    """
    if len(piles) > H:
        # 不可能实现
        return -1
    # 得到每堆香蕉的最大数目
    num_max = max(piles)
    for speed in range(1, num_max + 1):
        # 线性遍历循环
        if CanFinish(piles, speed, H):
            return speed
    return num_max


def minEatingSpeed_v2(piles, H):
    """
    @brief      香蕉问题(二分)

    @details    https://leetcode-cn.com/problems/koko-eating-bananas/

    @param      piles: 香蕉堆数
                H: 时间

    @return     K: 最小速度
    """
    if len(piles) > H:
        # 不可能实现
        return -1
    # 得到每堆香蕉的最大数目
    num_max = max(piles)
    left = 1
    right = num_max
    while(left < right):
        mid = (left + right) // 2
        if CanFinish(piles, mid, H):
            right = mid
        else:
            left = mid + 1
    return left


if __name__ == '__main__':
    piles = [3, 6, 7, 11]
    H = 8
    re1 = minEatingSpeed_v1(piles, H)
    re1_ = minEatingSpeed_v2(piles, H)
    piles = [30, 11, 23, 4, 20]
    H = 5
    re2 = minEatingSpeed_v1(piles, H)
    re2_ = minEatingSpeed_v2(piles, H)
    piles = [30, 11, 23, 4, 20]
    H = 6
    re3 = minEatingSpeed_v1(piles, H)
    re3_ = minEatingSpeed_v2(piles, H)
    print(re1_, re2_, re3_)
