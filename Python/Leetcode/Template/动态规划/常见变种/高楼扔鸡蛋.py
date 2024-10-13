# -*- coding: utf-8 -*-
'''
@author:    Wengang.Zheng
@email:     zwg0606@gmail.com
@fielName:  高楼扔鸡蛋.py
@Time:      2021-05-24-22:37:12
@Des:       核心思路：从策略角度思考非常复杂，可通过动态规划进行“穷举
            1. 状态量：当前拥有的鸡蛋数/需要测试的楼层数
            2. 状态转移：根据鸡蛋碎/不碎取两种情况，取max（因为是最坏情况）
'''
