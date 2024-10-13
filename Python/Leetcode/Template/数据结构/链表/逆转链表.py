# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 逆转链表.py
@Time:   2021-01-03-16:32:24
@Des: 递归逆转链表
'''


class ListNode():
    """
    @brief      简单的链表类
    """

    def __init__(self, val):
        self.val = val
        self.next = None


def reverse(head):
    """
    @brief      递归反转链表
    """
    # base case
    if head.next is None:
        return head

    # reverse的作用在于反转部分链表,并返回头结点
    last = reverse(head.next)
    head.next.next = head
    head.next = None
    return last


def reverseN(head, n):
    """
    @brief      递归反转前N个节点
    """
    # base case:要记录最后的N+1的节点
    if n == 1:
        successor = head.next
        return head

    last = reverseN(head.next, n - 1)
    head.next.next = head
    # 反转之后的head节点需要与后面的节点连接
    head.next = successor
    return last


def reverseBetween(head, m, n):
    """
    @brief      递归反转区间内的节点

    @param      head: 头节点
                m: 反转区间开始点
                n: 反转区间结束点
    """
    # base case, 当m==1的时候退化到reverseN
    if m == 1:
        return reverseN(head, n)

    # 不断前进到反转区间头节点
    head.next = reverseBetween(head.next, m - 1, n - 1)
    return head
