from collections import deque


def BFS(root, target_node):
    q = deque()
    visited = set()
    level = 0
    q.append(root)

    while len(q):
        size = len(q)
        level += 1
        for _ in range(size):
            node = q.popleft()
            if node == target_node:
                # do something here
                pass
            for neighbour in node.neighbours:
                if neighbour not in visited:
                    visited.add(neighbour)
                    q.append(neighbour)


def DFS(root, visited, target_node):
    if root is None:
        return

    if root == target_node:
        # do something
        return

    if root in visited:
        return

    visited.add(root)
    for neighbour in root.neighbours:
        DFS(neighbour, visited, target_node)
