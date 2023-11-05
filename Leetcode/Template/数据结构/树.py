# -*- coding: utf-8 -*-
'''
@author:    Wengang.Zheng
@email:     zwg0606@gmail.com
@fielName:  树.py
@Time:      2021-04-18-17:47:18
@Des:       树常见的使用方式和数据结构
'''


class Node(object):
    '''
    节点构建
    '''
    def __init__(self, value=-1, lchild=None, rchild=None):
        self.value = value
        self.lchild = lchild
        self.rchild = rchild


class Tree(object):
    def __init__(self, values):
        '''
        初始化树
        '''
        self.root = self.build(None, 0, values)

    def build(self, node, idx, values):
        '''
        迭代建树
        '''
        if idx >= len(values):
            return
        node = Node(values[idx])
        lchild = self.build(node.lchild, idx * 2 + 1, values)
        rchild = self.build(node.rchild, idx * 2 + 2, values)
        node.lchild = lchild
        node.rchild = rchild
        return node


class BST_Tree(object):
    def __init__(self, root):
        '''
        初始化BST_Tree
        '''
        self.root = root

    @classmethod
    def build_frome(cls, node_list):
        '''
        通过node_list构建BST
        '''
        pass


def preorder(node):
    '''
    前序遍历
    '''
    if node is None:
        return

    print(node.value)
    preorder(node.lchild)
    preorder(node.rchild)


def midorder(node):
    '''
    中序遍历
    '''
    if node is None:
        return

    midorder(node.lchild)
    print(node.value)
    midorder(node.rchild)


def postorder(node):
    '''
    后序遍历
    '''
    if node is None:
        return

    postorder(node.lchild)
    postorder(node.rchild)
    print(node.value)

# TODO 遍历方式：迭代
# 二叉查找树（BST）
# 平衡二叉树（AVL）


if __name__ == '__main__':
    # test for traverse a tree
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    print("values:", values)
    # build a tree
    tree = Tree(values)
    # preorder`
    print("order_pre:")
    order_pre = preorder(tree.root)
    # midorder
    print("order_mid:")
    order_mid = midorder(tree.root)
    # postorder
    print("order_post:")
    order_post = postorder(tree.root)
