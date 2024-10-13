# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 回文链表.py
@Time:   2021-01-31-16:32:05
@Des:
说明: 反转链表在于链表无法反向遍历
思路:
+ 构建反转链表,判断原始链表和反转链表是否一致,时间复杂度和空间复杂度均为O(n)
+ 递归进行倒序遍历,时间复杂度和空间复杂度均为O(n)
+ 双指针
'''


class ListNode():
    """
    @brief      链表类

    @details    简单表示链表类
    """
    def __init__(self, next_node, value):
        self.next_node = next_node
        self.value = value


# 递归进行倒序遍历
def isPalindrome(head):
    """
    @brief      判断回文链表：递归实现

    @details    https://leetcode-cn.com/problems/palindrome-linked-list/

    @param      head(ListNode): 头结点

    @return     res(bool): 是否为回文链表
    """
    global left
    left = head
    import copy
    r_left, res = traverse(left, copy.deepcopy(left))
    return res


def traverse(left, right):
    if right is None:
        return left, True

    # 递归到尾节点
    left, res = traverse(left, right.next_node)

    res = res and (right.val == left.val)
    # 判断下一个节点
    left = left.next_node

    return left, res


# 双指针先找到中间点后反转后半链表，降低空间复杂度为O(1)
def get_mid(head):
    slow = fast = head
    while (fast is not None and fast.next_node is not None):
        slow = slow.next_node
        fast = fast.next_node.next_node

    # 如fast未指向None，则说明为奇数链表，slow需要前进
    if fast is not None:
        slow = slow.next_node
    return slow


def reverse(head):
    pre = None
    cur = head

    while (cur is not None):
        next_node = cur.next_node
        cur.next_node = pre
        pre = cur
        cur = next_node

    return pre


def isPalindromev2(head):
    """
    @brief      判断回文链表：快慢指针优化

    @details    https://leetcode-cn.com/problems/palindrome-linked-list/

    @param      head(ListNode): 头结点

    @return     res(bool): 是否为回文链表
    """
    left = head
    slow = get_mid(head)
    right = reverse(slow)

    # p,q保存前置指针，目的是为了恢复原先的链表结构
    p = left
    q = right
    res = True

        if left.value != right.value:
            res = False

        p = left
        q = right
        left = left.next_node
        right = right.next_node

    if left is not None:
        # 奇数
        p = left

    # 恢复原先的链表结构
    p.next_node = reverse(q)
    return res
