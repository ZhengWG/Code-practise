# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 图.py
@Time:   2023-11-04-01:02:11
@Des:
'''


from typing import List


class Solution1:
    """
    @brief      0->n-1节点的所有可能的路径

    @details    https://leetcode.cn/problems/all-paths-from-source-to-target/
    """
    def __init__(self):
        # 记录所有结果
        self.res = []

    def allPathSourceTarget(self, graph: List[List[int]]):
        # 临时路径
        path = []
        # 有向无环图无需visited数组
        n = len(graph)
        traverse(self, graph, 0, n, path)

    def traverse(self, graph, i, n, path):
        """
        @brief      回溯
        """
        path.append(i)
        if i == n-1:
            self.res.append(path.copy())

        for j in graph[i]:
            traverse(graph, j, n, path)
        path.pop()


class Solution2:
    """
    @brief      判定二分图

    @details    https://leetcode.cn/problems/is-graph-bipartite/
    """
    def __init__(self):
        self.ok = True
        self.color = []
        self.visited = []

    def judge_binary_graph(self, graph: List[List[int]]):
        n = len(graph)
        self.color = [False] * n
        self.visited = [False] * n

        # !! 可能存在多个子图，所以对每个节点做判断
        for idx in range(len(graph)):
            if not self.ok:
                return self.ok
            if not self.visited[idx]:
                self.traverse(graph, idx)
        return self.ok

    def traverse(self, graph, i):
        # 回溯判定
        if not self.ok:
            return

        self.visited[i] = True
        for j in graph[i]:
            if self.visited[j]:
                # 已被访问，判断是否为不同类
                if self.color[i] != self.color[j]:
                    self.ok = False
                    return
            else:
                self.color[j] = !self.color[i]
                self.traverse(self, graph, j)


class Solution3:
    def __init__(self):
        pass

    def build_graph(self, courses: int, require_list: List[List[int]]):
        self.graph = [[] for _ in range(courses)]

        for edge in require_list:
            _from, _to = edge
            self.graph[_from].append(_to)

    def canFinish(self, courses: int, require_list: List[List[int]]):
        """
        @brief      判断图中是否有环

        @details    https://leetcode.cn/problems/course-schedule/
        """
        self.build_graph(courses, require_list)
        self.can_finish = True
        self.visited = [False] * courses  # 记录是否被访问
        self.op_path = [False] * courses  # 记录回溯路径

        for s in self.graph:
            if not self.can_finish:
                return self.can_finish
            if not self.visited[s]:
                self.traverse(s)

        return self.can_finish

    def topo_sort(self, courses: int, require_list: List[List[int]]):
        """
        @brief      拓扑排序：本质上为后序遍历的逆序输出

        @details    https://leetcode.cn/problems/course-schedule-ii/
        """
        self.build_graph(courses, require_list)
        self.can_finish = True
        self.visited = [False] * courses  # 记录是否被访问
        self.op_path = [False] * courses  # 记录回溯路径
        self.post_order = []  # 记录后续遍历结果

        for s in self.graph:
            if not self.can_finish:
                return []
            if not self.visited[s]:
                self.traverse(s, True)
        if not self.can_finish:
            return []

        self.post_order.reverse()
        return self.post_order

    def traverse(self, s, record=False):
        if self.op_path[s]:
            self.can_finish = False
            return
        if self.visited[s] or not self.can_finish:
            return

        self.visited[s] = True
        self.op_path[s] = True
        for j in self.graph[s]:
            self.traverse(j)
        if record:
            self.post_order.append(s)
        self.op_path[s] = False

class Solution4:
    """
    获取关键路径
    """
    def get_critical_path(self, ALGraph G):
        """
        G为邻接表存储的有向图，输出G的各项关键活动：事件最早发生的事件=最晚发生的事件
        参考：https://blog.csdn.net/m0_46518461/article/details/109725629
        Des：
         1. 获取拓扑排序
         2. 正向排序得到各个事件的最早发生事件
         3. 逆向排序得到各个事件的最晚发生事件
         4. 基于最早最晚事件判断是否为关键活动
        """
        # 拓扑排序
        if (!get_topo(G, topo)):
            return

        # 按照正向拓扑排序得到每个事件的最早发生时间
        for i in range(n):
            ve[i] = 0  # 初始化
        for i in range(n):
            node_idx = topo[i] # 获取拓扑排序顶点
            node = G[node_idx]
            while node is not None:
                j = node.next_idx
                if ve[j] < ve[i] + node.weight:
                    ve[j] = ve[i] + node.weight
                node = G[j]

        # 按照逆向拓扑排序得到每个事件的最晚发生时间
        for i in range(n):
            vl[i] = ve[n-1]
        for i in range(n-1, -1, -1):
            # 逆向拓扑
            node_idx = topo[i]
            node = G[node_idx]
            while node is not None:
                j = node.next_idx
                if vl[node_idx] > vl[j] - node.weight:
                    vl[node_idx] = vl[j] - node.weight
                node = G[j]

        # 判断活动是否为关键活动: ve[i] + weight = vl[i]
        for i in range(n):
            node = G[i]
            while node is not None:
                j = node.next_idx
                e = ve[i]
                l = vl[j] - node.weight
                if e == l:
                    print(f"关键路径：{{G[i].data}/{G[j].data}")
                node = G[j]
