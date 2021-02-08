# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: LRU算法.py
@Time:   2020-12-20-17:04:54
@Des:
简述LRU缓存机制:存储容量固定的数据结构,按照数据访问顺序排序,当数据量达到存储容量的时候会删除最早存储(访问)的数据,意味着如果访问到已有数据的话需要更新数据顺序.
关键方法:
+ put:存入数据,为最新访问顺序;如果数据结构达到容量上限则去除最早的数据
+ get:获取数据,更新数据为最新访问数据
算法思路:
要求put,get方法都是O(1)时间复杂度,所以采用字典数据结构;但是需要保证有序,所有本质上是有序字典数据结构,或者为hash双链表结构
'''


from collections import OrderedDict


class LRUCache:
    """
    @brief      直接基于有序字典OrderedDict实现
    """
    def __init__(self, capacity: int):
        # cache容量
        self.capacity = capacity
        # 有序字典存储访问顺序
        self.visited = OrderedDict()

    def get(self, key: int):
        """
        @brief      获取数据,并且更新数据到最后
        @return     return: value(int)
        """
        if key not in self.visited:
            return -1
        self.visited.move_to_end(key)
        return self.visited[key]

    def put(self, key: int, value: int):
        """
        @brief      存入数据,并更新到最后;如果超出容量,则弹出最旧的数据
        """
        if key not in self.visited and len(self.visited) == self.capacity:
            # 达到上限:去除最旧的数据,默认弹出最新的数据
            self.visited.popitem(last=False)

        self.visited[key] = value
        self.visited.move_to_end(key)


# TODO: 底层实现hash双链表结构: hash + 双链表
