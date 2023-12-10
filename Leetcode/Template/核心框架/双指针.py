# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 双指针.py
@Time:   2020-12-13-16:34:15
@Des:
'''


class ListNode():
    """
    @brief      简单链表类
    """
    def __init__(self, value, next_node=None):
        self.value = value
        self.next_node = next_node


def hasCycle(head):
    """
    @brief      判断链表是否存在环
    """
    # 初始化快慢指针
    fast = slow = head

    while fast is not None and fast.next_node is not None:
        fast = (fast.next_node).next_node
        slow = slow.next_node

        # 被套圈则说明有环
        if fast == slow:
            return True

    return False


def detectCycle(head):
    """
    @brief      找到有环链表的起始位置
    """
    # 初始化快慢指针
    fast = slow = head

    # 判断链表是否存在环
    while fast is not None and fast.next_node is not None:
        fast = (fast.next_node).next_node
        slow = slow.next_node

        # 相遇说明有环退出:此时相遇点和head点距离环起始点距离一致
        if fast == slow:
            break

    if fast is None or fast.next_node is None:
        return -1

    # 各自从head和相遇点等速度出发
    slow = head
    while slow != fast:
        slow = slow.next_node
        fast = fast.next_node


def get_sum(nums, target):
    """
    @brief      返回两数之和,假设存在唯一数对

    @details    https://leetcode-cn.com/problems/two-sum/

    @param      nums:输入数组,target:sum目标值

    @return     [int, int]:返回两数之和坐标
    """
    # 初始化指针
    left = 0
    right = len(nums) - 1

    while left < right:
        _sum = nums[left] + nums[right]
        if _sum > target:
            right -= 1
        elif _sum < target:
            left += 1
        elif _sum == target:
            return [left, right]
    return


def reverse(a_s):
    """
    @brief      反转数组,空间复杂度O(1),时间复杂度O(n)

    @details    https://leetcode-cn.com/problems/reverse-string/

    @param      a_s:输入数组/字符

    @return     list/str:反转后的数组/字符
    """
    # 初始化指针
    left = 0
    right = len(a_s) - 1

    while (left < right):
        # 替换两端value
        tmp = a_s[left]
        a_s[left] = a_s[right]
        a_s[right] = tmp
        right -= 1
        left += 1
    return a_s


if __name__ == '__main__':
    # test1
    nums = [2, 7, 11, 15]
    target = 9
    out = get_sum(nums, target)
    assert out == [0, 1], '{} != [0, 1]'.format(out)
    # test2
    a_s = ["h", "e", "l", "l", "o"]
    out = reverse(a_s)
    assert out == ["o", "l", "l", "e", "h"], '{} != ["o", "l", "l", "e", "h"]'.format(out)

    print("All test passed!")
