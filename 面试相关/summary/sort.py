# -*- coding: utf-8 -*-
'''
@Software  : EMACS
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Time: 2020-06-14-15:43:13
'''

import random
import time
import copy

MAX_NUM = 99999999


def swap(nums, i, j):
    tmp = nums[i]
    nums[i] = nums[j]
    nums[j] = tmp

def partial(nums, left, right):
    pivot = nums[right]
    idx = left
    while(idx < right):
        if nums[idx] <= pivot:
            swap(nums, idx, left)
            left += 1
        idx += 1
    swap(nums, left, right)
    return left


def quick_sort(nums, left, right):
    if left >= right:
        return
    idx_pivot = partial(nums, left, right)
    quick_sort(nums, left, idx_pivot-1)
    quick_sort(nums, idx_pivot, right)
    return


def heap_sink(heap, heap_size, parent_index):
    """
    最大堆-下沉算法
    """
    child_idx = 2 * parent_index + 1
    pivot = heap[parent_index]

    while (child_idx < heap_size):
        if child_idx + 1 < heap_size and  heap[child_idx] < heap[child_idx + 1]:
            child_idx += 1
        if pivot > heap[child_idx]:
            break

        # 交换父子节点
        heap[parent_index] = heap[child_idx]
        parent_index = child_idx
        child_idx = parent_index * 2 + 1

    heap[parent_index] = pivot


def heap_sort(nums):
    """
    堆排序
    """
    n = len(nums)

    # 构建堆：从后往前构建堆
    for i in range(n-1, -1, -1):
        heap_sink(nums, n, i)

    # 排序:依次取堆顶点后重新调整堆
    for i in range(n-1, -1, -1):
        nums[0], nums[i] = nums[i], nums[0]
        heap_sink(nums, i, 0)


def generate_random_seq(num):
    out = []
    for _ in range(num):
        out.append(random.randrange(MAX_NUM))
    return out


def main():
    nums = generate_random_seq(1000)
    nums_2 = copy.deepcopy(nums)
    nums_3 = copy.deepcopy(nums)
    t1 = time.time()
    quick_sort(nums, 0, len(nums)-1)
    t2 = time.time()
    print("time for quick_sort:", t2-t1)
    t1 = time.time()
    heap_sort(nums_2)
    t2 = time.time()
    print("time for heap_sort:", t2-t1)
    t1 = time.time()
    nums_3.sort()
    t2 = time.time()
    print("time for python_sort:", t2-t1)

main()
