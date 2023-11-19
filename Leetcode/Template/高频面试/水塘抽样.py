# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 水塘抽样.py
@Time:   2021-01-31-12:59:55
@Des:
给定未知长度链表,设计算法只能遍历一次,随机返回链表中的一个节点,代码简单,关键在于思路
注意: 随机为均匀随机,要求不存在统计意义上的偏差
思路:
+ 时间复杂度O(n)算法,对于i个元素,选中i个元素的概率为1/i,遇到第i+1个元素替换的概率为1-1/(i+1),以此类推,则最后选中i个元素的概率为1/n,不存在统计意义上的偏差
+ TODO:基于几何分布(geometric distribution),时间复杂度为O(k+klog(N/k))
+ TODO:Fisher-Yates洗牌算法,算法的思路为随机访问元素,只支持对数组这类随机存储的数据结构
+ TODO:拓展应用:为每个元素关联随机数,维护K的二叉堆,算法复杂度为O(nlogk),该算法的意义在于支持"加权随机"
'''


class ListNode():
    """
    @brief      链表类

    @details    简单表示链表类
    """
    def __init__(self, next_node, value):
        self.next_node = next_node
        self.value = value


def getRandom(head):
    """
    @brief      随机返回节点

    @details    保证随机概率

    @param      head(ListNode): 链表头节点

    @return     res(float): 返回的随机节点value值
    """
    p = head

    import random
    idx = 0
    res = None
    while(p is not None):
        num = random.randint(0, idx)

        # 概率:1/(idx + 1)
        if num == 0:
            res = p.value
        p = p.next_node
        idx += 1
    return res


def getRandomK(head, k):
    """
    @brief      随机返回K个节点

    @details    类似返回单个节点,只是概率变大K倍

    @param      head(ListNode): 链表头节点
                k(int): 返回节点数目

    @return     res(list): 返回的k个随机节点value值
    """
    p = head

    import random
    res = [None] * k
    # 默认选上前k个元素
    for idx in range(k):
        res[idx] = p.value
        p = p.next_node

    while(p is not None):
        num = random.randint(0, idx)

        # 概率:k/(idx + 1)
        if num < k:
            res[num] = p.value
        p = p.next_node
        idx += 1

    return res
