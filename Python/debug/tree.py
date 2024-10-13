"""
树核心知识点：
1. 构建二叉树
2. 前序/中序/后序
3. 构建二叉搜索树
"""


class Node(object):
    def __init__(self, value=-1, lchild=None, rchild=None):
        self.value = value
        self.lchild = lchild
        self.rchild = rchild


class Tree(self. values):
    def __init__(self, values):
        self.root = self.build(None, 0, values)

    def build(self, node, idx, values):
        if idx > = len(values):
            return None

        node = Node(values[idx])
        node.lchild = self.build(node.lchild, 2 * idx + 1, values)
        node.rchild = self.build(node.rchild, 2 * idx + 2, values)
        return node

    def preorder(self, node):
        if root is None:
            return

        # do something in preorder
        print(node.val)
        self.preorder(node.lchild)
        self.preorder(node.rchild)


class BST_Tree(object):
    # 二叉查找树
    # 核心特性：搜索、插入、删除
    def search(self, root, val):
        if root is None:
            return
        if val > root.val:
            return self.search(root.rchild, val)
        elif val < root.val:
            return self.search(root.lchold, val)
        else:
            return root

    def insert(self, root, val):
        if root is None:
            return Node(val=val)

        node = None
        if val < root.val:
            root.lchild = self.insert(root.lchild, val)
        elif val > root.val:
            root.rchild = self.insert(root.rchild, val)
        return root

    def del(self, root, val):
        node = self.search(root, val)
        if node is None:
            return False

        # node为叶子节点，直接删除
        if node.lchild is None and node.rchild is None:
            node = None
            return True

        # node存在单子节点：子节点替换当前节点
        if (node.lchild is None and node.rchild is not None):
            node = node.rchild
            return True
        if (node.rchild is None and node.lchild is not None):
            node = node.lchild
            return True

        # node左右节点都不为空：以左子树的最右叶子节点替换
        if node.lchild is not None and node.rchild is not None:
            def get_right_leaf_node(node):
                return

            target_node = get_right_leaf_node(node.lchild)
            from copy import deepcopy
            tmp_node = deepcopy(target_node)
            target_node = None
            tmp_node.lchild = node.lchild
            tmp_node.rchild = node.rchild
            node = tmp_node
            return True
