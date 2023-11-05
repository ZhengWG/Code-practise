# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 优先级队列.py
@Time:   2020-12-20-17:51:09
@Des:
基于数组实现最大堆,基本操作为insert, delMax
'''


# 采用数组存储二叉树
def parent(root: int):
    """
    @brief      返回根节点index
    """
    return root // 2


def left(root: int):
    """
    @brief      返回左孩子索引
    """
    return root * 2


def right(root: int):
    """
    @brief      返回右孩子索引
    """
    return root * 2 + 1


class MaxPQ:
    """
    @brief      建立最大堆
    """
    def __init__(self):
        # 不用0索引,故初始化为None
        self.pq = [None]
        self.num = 0

    def max(self):
        """
        @brief      返回队列最大元素
        @return     max_value: int
        """
        return self.pq[1]

    def insert(self, value):
        """
        @brief      加入新元素,并且调整最大堆:加到最后然后上浮
        """
        # 更新数列数目
        self.num += 1
        self.pq.append(value)
        self.swim(self.num)

    def delMax(self):
        """
        @brief      删除最大元素:即idx=1的元素
        @details    交换第一个元素和最后一个元素,然后删除最后一个元素,进行下沉
        @return     max_value: int
        """
        # 交换元素
        max_value = self.pq[1]
        self.exch(1, self.num)
        # 删除最后一个元素
        del self.pq[self.num]
        self.num -= 1
        # 进行下沉
        self.sink(1)
        return max_value

    def swim(self, k: int):
        """
        @brief      上浮第k个元素,以维护最大堆性质
        """
        # 循环不断上浮,如果到堆顶或者k个元素小于其父节点则不再上浮
        while (k > 1 and self.pq[k] > self.pq[parent(k)]):
            # 交换父子节点
            self.exch(k, parent(k))
            # 更新k
            k = parent(k)

    def sink(self, k: int):
        """
        @brief      下沉第k个元素,以维护最大堆性质
        """
        # 循环不断下沉,如果左节点超出则说明已到堆底
        while (left(k) <= self.num):
            # 判断左右节点和当前节点大小:找更小的节点下沉
            sinked_idx = 0
            if self.pq[k] > self.pq[left(k)]:
                sinked_idx = left(k)

            # 如果右节点存在并且大于左节>
            if right(k) <= self.num and self.pq[left(k)] > self.pq[right(k)] and self.pq[right(k)] < self.pq[k]:
                sinked_idx = right(k)

            if not sinked_idx:
                # 如果左右节点都大于当前节点则直接break
                break
            # 下沉操作:更换节点内容并更新k
            self.exch(sinked_idx, k)
            k = sinked_idx

    def exch(self, i: int, j: int):
        """
        @brief      交换数组的两个元素
        """
        tmp = self.pq[i]
        self.pq[i] = self.pq[j]
        self.pq[j] = tmp


if __name__ == '__main__':
    # test for MaxPQ
    maxpq = MaxPQ()
    import random
    for _ in range(10):
        inserted_num = random.randint(1, 100)
        maxpq.insert(inserted_num)

    assert maxpq.max() == max(maxpq.pq[1:]), '{}!={}'.format(maxpq.max, max(maxpq.pq))
    maxpq.delMax()
    assert maxpq.max() == max(maxpq.pq[1:]), '{}!={}'.format(maxpq.max, max(maxpq.pq))
