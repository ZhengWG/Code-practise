# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: Union-Find.py
@Time:   2021-02-07-23:44:37
@Des:
图的动态流动性:n个节点,核心操作:连通点&&判断点是否连通
连通语义:自反性/对称性/传递性
关键点:
+ 本质为森林数据结构(若干树)
+ 为了保障数据结构的高效需要考虑:
    + 保障树结构的平衡
    + 需要进行树结构的压缩
'''
from copy import deepcopy


class UF():
    """
    @brief      Union-Find类

    @details    关键方法: 将两个节点相连+判断两个节点是否相连
    """
    def __init__(self, n):
        """
        @brief      初始化图

        @details    所有图节点初始化父节点都为自身

        @param      n(int): 图节点数目
        """
        self.num = n
        # 父节点
        self.parent = [-1] * n
        # 记录根节点的树的节点数目
        self.size = [-1] * n
        for i in range(n):
            self.parent[i] = i
            self.size[i] = 1

    def union(self, p, q):
        """
        @brief      将两个节点相连

        @param      p(int): 节点p index
                    q(int): 节点q index
        """
        rootp = self._find(p)
        rootq = self._find(q)

        if rootq == rootp:
            return

        # 合并树
        # 小树接到大树下面,保持平衡
        if self.size[rootp] < self.size[rootq]:
            self.parent[rootq] = rootp
            self.size[rootp] += self.size[rootq]
        else:
            self.parent[rootp] = rootq
            self.size[rootq] += self.size[rootp]
        # 连通分量-1
        self.num -= 1

    def _find(self, x):
        """
        @brief      返回某个节点的根节点

        @param      x(int): 需要返回的节点的index

        @return     x(int): 返回的根节点的index
        """
        while (self.parent[x] != x):
            # 进行路径压缩
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def connected(self, p, q):
        """
        @brief      判断两个节点是否相连

        @param      p(int): 节点p index
                    q(int): 节点q index

        @return     res(bool): 节点p, q是否相连
        """
        res = (self._find(p) == self._find(q))
        return res

    def count(self):
        """
        @brief      返回图的连通分量数目

        @return     num(int): 当前图的连通分量数目
        """
        num = self.num
        return num


def solve(area):
    """
    @brief      被围绕区域

    @details    https://leetcode-cn.com/problems/surrounded-regions/

    @param      area(list): 包含位置信息的矩阵区域

    @return     res(list): 去除围绕位置的矩阵
    """
    res = deepcopy(area)

    if not len(area):
        return area

    m = len(area)
    n = len(area[0])

    # 余留1个位置作为虚拟祖先(根节点)dummy
    uf = UF(m * n + 1)
    dummy = m * n
    # 将首列和末列的'O'和dummy相连
    for i in range(m):
        if area[i][0] == 'o':
            uf.union(i * n, dummy)
        if area[i][n - 1] == 'o':
            uf.union(i * n + n - 1, dummy)

    # 将首行和末行的'o'与dummy相连
    for j in range(n):
        if area[0][j] == 'o':
            uf.union(j, dummy)
        if area[m - 1][j] == 'o':
            uf.union(n * (m - 1) + j, dummy)

    # 构建方向数组:常用于上下左右搜索
    d = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    for i in range(1, m):
        for j in range(1, n):
            for k in range(4):
                if area[i][k] == 'o':
                    x = d[k][0]
                    y = d[k][1]
                    if area[x][y] == 'o':
                        uf.union(x * n + y, i * n + j)

    # 所有父节点非dummy的'o'都说明被围住了
    for i in range(m):
        for j in range(n):
            if not uf.connected(dummy, i * n + j):
                res[i][j] = 'x'

    return res


def equationPossible(equations):
    """
    @brief      判定合法等式

    @details    https://leetcode-cn.com/problems/satisfiability-of-equality-equations/

    @param      equations(list): 一系列等式

    @return     res(bool): 是否都为合法等式
    """
    # 先让26个字母各自形成自身父节点
    uf = UF(26)
    for eq in equations:
        if eq[1] == '=':
            # 相等的字母进行连通
            x = eq[0]
            y = eq[3]
            uf.union(ord(x) - ord('a'), ord(y) - ord('a'))

    for eq in equations:
        if eq[1] == '!':
            # 如果相等关系成立则为逻辑冲突
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            if uf.connected(x, y):
                return False
    return True


if __name__ == '__main__':
    input1 = [['x', 'x', 'x', 'x'],
              ['x', 'o', 'o', 'x'],
              ['x', 'x', 'o', 'x'],
              ['x', 'o', 'x', 'x']]

    ans1 = [['x', 'x', 'x', 'x'],
            ['x', 'x', 'x', 'x'],
            ['x', 'x', 'x', 'x'],
            ['x', 'o', 'x', 'x']]

    output1 = solve(input1)
    assert output1 == ans1

    input2 = ['a==b', 'b!=a']
    input3 = ['b==a', 'a==b']

    output2 = equationPossible(input2)
    output3 = equationPossible(input3)

    assert output2 == False and output3 == True
    print("All test passed!")
