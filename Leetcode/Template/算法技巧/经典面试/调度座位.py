# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 调度座位.py
@Time:   2021-01-31-00:45:52
@Des:
说明: 单个学生进入的时候,需要最大化其和最近其他人的距离;如果存在多个位置,则安排其在索引最小的位置
核心思路:
+ 将学生位置抽象为线段端点,选位置问题抽象为选取最大线段的中点
+ 最值问题常用数据结构:二叉堆/平衡二叉树(AVL)
'''


class TreeSet():
    """
    @brief      AVL树

    @details    TODO: 具体实现
    """
    def __init__(self, cmp_f):
        """
        @brief      初始化AVL树

        @param      cmp_f: 比较函数
        """
        pass

    def add(self, node):
        """
        @brief      添加元素
        """
        pass

    def remove(self, node):
        """
        @brief      删除元素
        """
        pass

    def last(self):
        """
        @brief      返回最大数据
        """
        pass

    def first(self):
        """
        @brief      返回最小数据
        """
        pass


class ExamRoom():
    """
    @brief      安排座位类

    @details    https://leetcode-cn.com/problems/exam-room/
    """
    def __init__(self, n):
        """
        @brief      初始化位置
        """
        self.num = n
        # 存储起始点->线段
        self.startmap = {}
        # 存储终点->线段
        self.endmap = {}
        # 储存线段长度AVL
        self.pq = TreeSet(cmp_f=self.compare_dis)

        # 初始化位置:-1,n
        p_init = [-1, n]
        self._addInterval(p_init)

    @staticmethod
    def distance(a):
        """
        @brief      返回线段中点长度: 最大化距离
        """
        # return a[1] - a[0] - 1
        left = a[0]
        right = a[1]
        return (right - left) / 2

    @staticmethod
    def compare_dis(a, b):
        """
        @brief      比较线段长度: 相等则返回idx更小的
        """
        disa = ExamRoom.distance(a)
        disb = ExamRoom.distance(b)

        if disa == disb:
            return b[0] - a[0]
        return disa - disb

    def _addInterval(self, p):
        """
        @brief      插入位置(端点)

        @param      p(list): 需要插入的线段点,包含起点/终点
        """
        self.pq.add(p)
        self.startmap[p[0]] = p
        self.startmap[p[1]] = p

    def _removeInterval(self, p):
        """
        @brief      删除位置(端点)

        @param      p(list): 需要删除的线段点,包含起点/终点
        """
        self.pq.remove(p)
        self.startmap.remove(p[0])
        self.startmap.remove(p[1])

    def seat(self):
        """
        @brief      安排位置

        @details    需要最大化其和最近其他人的距离

        @return     idx(int): 位置序号
        """
        # 得到最长长度的线段
        longest = self.pq.last()
        start = longest[0]
        end = longest[1]

        if start == -1:
            seat = 0
        elif end == self.num:
            seat = self.num - 1
        else:
            # 返回中点
            seat = (end - start) / 2 + start

        # 删除最长线段,添加分割后的线段
        left = [x, seat]
        right = [seat, y]
        self._removeInterval(longest)
        self._addInterval(left)
        self._addInterval(right)
        return seat

    def leave(self, seat):
        """
        @brief      删除对应位置数据

        @param      p(int): 需要删除的位置序号
        """
        left = self.endmap[seat]
        right = self.startmap[seat]

        # 插入合并左右线段并删除
        merged = [left[0], right[1]]
        self._removeInterval(left)
        self._removeInterval(right)
        self._addInterval(merged)
