# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: k个一组翻转链表.py
@Time:   2021-01-17-16:27:17
@Des:
关键在于递归返回k个节点反转后的节点
'''


class ListNode():
    """
    @brief      简单链表类
    """
    def __init__(self, value, next_node = None):
        self.value = value
        self.next = next_node


def reverse(a, b):
    """
    @brief      翻转[a, b)的链表:注意为左闭右开
    """
    pre = None
    cur = nxt = a

    while (cur != b):
        # 迭代翻转
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt

    # 返回翻转后的头节点
    return pre


def reverseKGroup(head, k):
    """
    @brief      迭代翻转k个子链表节点

    @details    https://leetcode-cn.com/problems/reverse-nodes-in-k-group/

    @param      head: 链表子节点
                k: 迭代的子链表数目

    @return     翻转后的头节点
    """
    if head is None:
        return None

    a = b = head

    # 判断是否需要翻转
    for i in range(k):
        if b.value is None:
            # 如果不足k直接返回head
            return head
        b = b.next

    # 翻转当前k个节点, 并得到翻转后的头节点
    newhead = reverse(a, b)
    # 对剩下的链表部分进行递归翻转
    a.next = reverseKGroup(b, k)
    #　返回当前翻转的头节点
    return newhead
