# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 回溯.py
@Time:   2020-12-13-01:04:14
@Des:
回溯本质是决策树的遍历过程
需要思考的三个问题:
+ 路径: 已经做出的选择
+ 选择列表: 当前可做的选择
+ 结束条件: 到达决策树底层,无法再做选择的条件


算法框架核心:在递归调用之前[做选择],在递归之后撤销选择
result = []
def backtrack(路径, 选择列表):
    if 满足条件:
        result.add(路径)
        return
    for 选择 in 选择列表:
        做选择,并从选择列表从移除该选择
        backtrack(路径，选择列表)
　　　　撤销选择，并将该选择加入选择列表
回溯算法特点:暴力穷举,时间复杂度始终为:O(n!)
'''


from copy import deepcopy


res_permute = []
res_queens = []


def permute(nums):
    """
    全排列主函数：输入一组不重复的数字，返回他们的全排列
    """
    track = []
    backtrack(nums, track)
    res = deepcopy(res_permute)
    return res


def backtrack(nums, track):
    # 终止条件：track内数目==len(nums)
    if not len(track) ==  len(nums):
        res_permute.append(deepcopy(track))
        return

    for idx, _num in enumerate(nums):
        # 排除不合法的选择
        if _num in track:
            continue
        # 添加选择
        track.append(_num)
        backtrack(nums, track)
        # 撤销选择
        track.remove(_num)


def solveQueens(num):
    """
    八皇后问题:输入棋盘边长为num*num,返回所有合法的布置
    八皇后限制:周围一圈8个位置不能存在其他旗子
    """
    # *:无棋子,Q:有棋子
    # 二维数组初始化的时候*只会复制list对象,会有问题
    track = [['*' for col in range(num)] for row in range(num)]
    backtrack_queen(0, track)
    res = deepcopy(res_queens)
    return res


# 辅助判断函数
def is_valid_queen(track, row, col):
    """
    判断当前位置是否为符合要求的棋子位置
    """
    for shift_x in [-1, 0, 1]:
        for shift_y in [-1, 0, 1]:
            if shift_x == 0 and shift_y == 0:
                continue
            _row = shift_x + row
            _col = shift_y + col
            if _row < 0 or _row >= len(track):
                continue
            if _col < 0 or _col >= len(track[_row]):
                continue
            if track[_row][_col] == 'Q':
                return False
    return True


def backtrack_queen(num_row, track):
    # 终止条件:num达到了track行数
    if len(track) == num_row:
        _res = deepcopy(track)
        res_queens.append(_res)
        return

    # 对单行所有列进行布置
    num_col = len(track[num_row])
    for idx in range(num_col):
        if not is_valid_queen(track, num_row, idx):
            continue
        # 添加选择
        track[num_row][idx] = 'Q'
        backtrack_queen(num_row + 1, track)
        # 撤销选择
        track[num_row][idx] = '*'


if __name__ == '__main__':
    # test permute
    # res_permute = []
    # res = permute(list(range(1)))
    # print(res)
    # test solveQueens
    res_queens = []
    res = solveQueens(3)
    for r in res:
        print("##########")
        str_re = ''
        for row in r:
            for col in row:
                str_re += col
                str_re += ' '
            str_re += "\n"
        print(str_re)
