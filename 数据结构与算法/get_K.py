# -*- coding: utf-8 -*-
#!/usr/bin/env python3
'''
@Software  : EMACS
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Time:   2020.07.19
@Des:    取第k小数
'''
import copy
import time
import numpy as np


np.random.seed(66)


def normal_method(input_nums, k):
    """
    直接排序:nlogn
    """
    out_nums = copy.deepcopy(input_nums)
    start_time = time.time()
    out_nums.sort()
    out_num = out_nums[k-1]
    end_time = time.time()
    print("normal_method:", end_time - start_time)
    return out_num


def que_method(input_nums, k):
    """
    建立k的最大堆：记录当前最小的k个数
    单次调整堆的时间复杂度:log(k),总和时间复杂度:nlog(k)
    """
    out_nums = copy.deepcopy(input_nums)
    import heapq
    start_time = time.time()
    k_que = out_nums[:k]
    heapq._heapify_max(k_que)
    for num in out_nums[k:]:
        # 如果num小于当前的堆顶，则替换堆顶，并调整最大堆
        if num < k_que[0]:
            k_que[0] = num
            heapq._siftup_max(k_que, 0)
    out_num = k_que[0]
    end_time = time.time()
    print("que_methold:", end_time - start_time)
    return out_num


def main():
    max_num = int(10e10)
    nums = int(10e5)
    k = int(10e2)
    assert k <= nums
    input_nums = [np.random.randint(0, max_num) for _ in range(nums)]

    print("normal_method:", normal_method(input_nums, k))
    print("que_method:", que_method(input_nums, k))


if __name__ == '__main__':
    main()
