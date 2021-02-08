# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 单调队列.py
@Time:   2020-12-31-19:11:03
@Des:
此处单调队列相对狭义，只是维护了数组的单调递减的部分,而把中间内容删除
适用场景:解决滑动窗口的一系列问题
'''


from collections import deque


class MonotonicQueue():
    """
    @brief      构建单调队列数据结构
    """
    def __init__(self):
        self.queue = deque()

    def max(self):
        """
        @brief      返回最大值:O(1)
        """
        return self.queue[0]

    def pop(self, n):
        """
        @brief      去除对应的元素:仅删除队首(因为队首为最大元素)
        """
        if not len(self.queue) and self.queue[0] == n:
            self.queue.popleft()

    def push(self, n):
        """
        @brief      压入元素,并删除前面比自己小的元素:保持队列的单调性
        """
        while (len(self.queue) and self.queue[-1] < n):
            self.queue.pop()

        self.queue.append(n)


def maxslidingWindow(nums, k):
    """
    @brief      获取滑动窗口最大值(时间复杂度:O(n))

    @details    https://leetcode-cn.com/problems/sliding-window-maximum/

    @param      nums: 输入数组;k: 滑动窗口长度

    @return     res(list): 滑动窗口内最大值
    """
    # 构建单调队列数据结构
    window = MonotonicQueue()
    res = []

    for i in range(len(nums)):
        if i < k - 1:
            # 先填满window
            window.push(nums[i])
        else:
            window.push(nums[i])
            # 得到当前窗口的最大值:O(1)
            res.append(window.max())
            # 删除最前的元素:如果存在于队首的话,否则说明已经被删除
            window.pop(nums[i - k + 1])

    return res


if __name__ == '__main__':
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    res = maxslidingWindow(nums, k)
    assert res == [3, 3, 5, 5, 6, 7]
    print("Passed all test")
