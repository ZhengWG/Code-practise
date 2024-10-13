# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 素数个数.py
@Time:   2021-01-03-21:56:27
@Des: 高效得到区间内的素数个数
'''

import math


def isPrime(n):

    """
    @brief      判断是否为素数
    """
    if n < 0:
        raise ValueError(n)
    if n == 1 or n == 0:
        return False
    if n == 2:
        return True
    for i in range(2, int(math.sqrt(n) + 1)):
        if not n % i:
            return False
    return True


def countPrimes(n):
    """
    @brief      找到n内的素数

    @details    时间复杂度:O(NloglogN), Sieve of Eratosthenes
    """
    isPrim = [True] * n
    for i in range(2, int(math.sqrt(n) + 1)):
        if isPrim[i]:
            # 反向思路进行累计
            for j in range(i * i, n, i):
                # 从i*i开始遍历,单次叠加i
                isPrim[j] = False

    res = []
    for i in range(2, n):
        if isPrim[i]:
            res.append(i)
    return res


if __name__ == '__main__':
    n = 12
    res = countPrimes(n)
    assert res == [2, 3, 5, 7, 11]
    print("All test passed!")
