# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 去除有序数组重复元素.py
@Time:   2021-01-10-17:43:23
@Des:
时间复杂度:O(n)
空间复杂度:O(1)
'''


def deleteDuplicates(nums):
    """
    @brief       快慢指针
    """
    slow = fast = 0

    while(fast < len(nums)):
        if nums[slow] != nums[fast]:
            # 更新数组
            slow += 1
            nums[slow] = nums[fast]
        fast += 1

    return [nums[i] for i in range(slow + 1)]


if __name__ == '__main__':
    import random
    random.seed(10)
    nums = []
    for _ in range(1000):
        nums.append(random.randint(0, 10))
    nums.sort()
    nums_s = list(set(nums))
    assert deleteDuplicates(nums) == nums_s
    print("All test passed!")
