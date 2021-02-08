# -*- coding: utf-8 -*-
'''
@Author: Wengang.Zheng
@Email:  zwg0606@gmail.com
@Filename: 动态规划.py
@Time:   2020-12-13-01:03:31
@Des:
动态规划基本思路:
明确base case->明确状态->明确选择->定义dp数组/函数的定义
代码框架:
# 初始化 base case
dp[0][0][...] = base
# 进行状态转移
for 状态1 in 状态1的所有可选值:
    for 状态2 in 状态2的所有可选值:
        for ...
            dp[状态1][状态2][...] = 求最值(选择1, 选择2, ...  )
'''


# 斐波那契数列:
# 数列组成:arr[n] = arr[n-1] + arr[n-2], arr[0] = arr[1] = 1
def fib(num):
    """
    @brief      常规递归解法
                时间复杂度: O(2^n),存在2^n子问题(二叉树结构),子问题时间复杂度为加法操作O(1)
                算法问题:存在重复计算:f(n-1)需要计算f(n-2),f(n-3),f(n-2)又会重复计算f(n-3)
    """
    if num in (1, 2):
        return 1

    return fib(num - 1) + fib(num - 2)


def build_help(helper, num):
    if num < 0:
        return 0
    if num <= 1:
        return 1
    # 不再重复计算
    if helper[num - 1] is not None:
        return helper[num - 1]
    else:
        return build_help(helper, num - 1) + build_help(helper, num - 2)


def fib_with_helper(num):
    """
    @brief      构建备忘录(helper)辅助计算
                时间复杂度: O(n),进行了子树的剪枝操作,2^n->n,子问题时间复杂度为加法操作O(1)
                算法说明:自顶而上计算
    """
    # helper初始值:None
    helper = [None] * num
    return build_help(helper, num)


def fib_with_dp(num):
    """
    @brief      构建dp_table辅助计算
                时间复杂度:O(n),最基本的状态转移方程:f(n)=f(n-1)+f(n-2),n>2
                算法说明:自底而上计算
    """
    # 初始化前2初始值
    dp_table = [1] * 2
    if num <= 2:
        return dp_table[num - 1]

    for idx in range(num - 2):
        value = dp_table[idx - 1] + dp_table[idx - 2]
        dp_table.append(value)
    return dp_table[num - 1]


# 凑零钱问题
# 给定k个钱币,币值为c_1,...,c_k,要求给到总金额c_sum,求至少需要几枚硬币,如无法凑出,则返回-1,硬币可以重复使用
# 明确base case: amount(当前金额)=0 表示不需要硬币了;amount<0 表示无法完成
# 确定状态: 可变量为amount, 作为状态更新值
# 确定选择: 状态发生的原因:选择一个硬币值
# 确定dp函数/数组定义: dp函数参数值为状态更新值,其函数作用定义为:输入目标金额值:amount,得到需要的最少金币数


def coinchange(coins, amount):
    """
    @brief      直接递归暴力解法
                时间复杂度: 本质是N叉树迭代,层数为n^k,每层计算量为O(k),故时间复杂度为O(k*n^k),还是存在重复计算

    @param      coins(list): c_1,...,c_k 硬币币值
                amount(int): 目标金额
    """
    # 不断递归的dp函数
    def dp(n):
        # 确认边界: base case
        if n == 0:
            # 表示完成了
            return 0
        if n < 0:
            # 表示无法完成
            return -1
        # 初始化值
        res = float('INF')

        for coin in coins:
            # 子问题结果: 选择coin值, 状态值改变: n - coin
            subproblem = dp(n - coin)
            if subproblem == -1:
                # 表示无法完成
                continue
            # 当前值/子问题结果+1
            res = min(res, 1 + subproblem)

        return res if res != float('INF') else -1

    return dp(amount)


def coinchange_withmem(coins, amount):
    """
    @brief      带备忘解法
                时间复杂度: 剪枝后层数为O(n),每层计算量为O(k),故时间复杂度为O(k*n)

    @param      coins(list): c_1,...,c_k 硬币币值
                amount(int): 目标金额
    """
    # 初始化备忘: dict
    mem = dict()

    # 不断递归的dp函数
    def dp(n):
        # 确认边界: base case
        if n == 0:
            # 表示完成了
            return 0
        if n < 0:
            # 表示无法完成
            return -1
        # 查备忘,避免重复计算
        if n in mem:
            return mem[n]
        # 初始化值
        res = float('INF')

        for coin in coins:
            # 子问题结果: 选择coin值, 状态值改变: n - coin
            subproblem = dp(n - coin)
            if subproblem == -1:
                # 表示无法完成
                continue
            # 当前值/子问题结果+1
            res = min(res, 1 + subproblem)

        # 得到最终结果后, 更新备忘
        mem[n] = res if res != float('INF') else -1
        return res if res != float('INF') else -1

    return dp(amount)


def coinchange_iter(coins, amount):
    """
    @brief      递归解法
                思路: 维护dp表, dp[i]意味着目标值为i时需要的最小金币值
                时间复杂度: O(k*n)

    @param      coins(list): c_1,...,c_k 硬币币值
                amount(int): 目标金额
    """
    dp_table = [float('INF')] * (amount + 1)
    # dp_table[0]表示金额为0
    dp_table[0] = 0

    for i in range(amount + 1):
        # 遍历取金币
        for coin in coins:
            # 子问题状态值
            sub_amount = i - coin
            if sub_amount < 0:
                # 无法达到
                continue
            dp_table[i] = min(dp_table[i], 1 + dp_table[sub_amount])

    return dp_table[amount] if dp_table[amount] != float('INF') else -1
