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
        if self.visited[s] or not self.can_finish:
            return
        if self.op_path[s]:
            self.can_finish = False
            return

        self.visited[s] = True
        self.op_path[s] = True
        for j in self.graph[s]:
            self.traverse(j)
        if record:
            self.post_order.append(s)
        self.op_path[s] = False

