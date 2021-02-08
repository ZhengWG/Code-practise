# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 遍历.py
@Time:   2020-12-13-01:03:04
@Des:
简单谈谈一般遍历方法:
+ 数据
+ 链表
+ 二叉树
'''


def do_something(a):
    print(a)


def traverse_arr(list_arr):
    """
    @brief      数组遍历
    """
    for mem_arr in list_arr:
        # do something
        do_something(mem_arr)


class ListNode():
    """
    @brief      简单的链表类
    """

    def __init__(self, value, next_node=None):
        self.value = value
        self.next_node = next_node


def traverse_listnode_iter(head):
    """
    @brief      迭代遍历链表
    """
    node = head
    while node is not None:
        do_something(node.value)
        node = node.next_node


def traverse_listnode_recur(head):
    """
    @brief      递归遍历链表
    """
    if head is None:
        return
    do_something(head.value)
    traverse_listnode_recur(head.next_node)


class BiTreeNode():
    """
    @brief      简单的二叉树节点
    """

    def __init__(self, value, left_node=None, right_node=None):
        self.left_node = left_node
        self.right_node = right_node
        self.value = value


def traverse_bitreenode_recur(treenode):
    """
    递归遍历二叉树
    和节点遍历基本一致,区别是:next_node->left_node/right_node
    """
    if treenode is None:
        return

    do_something(treenode.value)

    traverse_bitreenode_recur(treenode.left_node)
    traverse_bitreenode_recur(treenode.right_node)


class TreeNode():
    """
    简单的N叉树节点
    """

    def __init__(self, value, childnodes=None):
        self.childnodes = childnodes
        self.value = value


def traverse_treenode_recur(treenode):
    """
    递归遍历N叉树
    """
    if treenode is None:
        return

    do_something(treenode.value)

    for childnode in treenode.childnodes:
        traverse_treenode_recur(childnode)
