# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: BFS.py
@Time:   2023-11-04-01:02:11
@Des:
+ BFS模板
'''


from typing import List, Set
from collections import deque


class ListNode:
    def __init__(self, val, neighbours=[]):
        self.val = val
        self.neighbours = neighbours


def BFS(root: ListNode, target: int):
    """
    @brief      广度遍历模板
    """
    q = deque()
    visited = set()
    q.append(root)
    step = 0

    while q:
        size = len(q)
        step += 1
        for i in range(size):
            cur_node = q.popleft()
            if cur_node.val == target:
                return step
            for neighbour in cur_node.neighbours:
                if neighbour not in visited:
                    visited.add(neighbour)
                    q.append(neighbour)

    return -1
