# -*- coding: utf-8 -*-
'''
@author:    Wengang.Zheng
@email:     zwg0606@gmail.com
@fielName:  区间调度问题.py
@Time:      2021-05-24-22:33:12
@Des:       区间调度问题：解决不相交子集
            核心：贪心算法，关键步骤：
            1. 区间集合中选择区间x，x为当前区域中结束最早的
            2. 把所有和x存在相交的区间去除
            3. 重复步骤1和2，直到集合为空为止。之前选出的子集集合即为最大不相交子集
'''
