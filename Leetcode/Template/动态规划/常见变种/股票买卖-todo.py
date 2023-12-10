# -*- coding: utf-8 -*-
'''
@author:    Wengang.Zheng
@email:     zwg0606@gmail.com
@fielName:  股票买卖.py
@Time:      2021-05-24-16:43:41
@Des:       股票买卖问题的核心在于：穷举状态选择
            for 状态1 in 状态1的所有取值：
                for 状态2 in 状态2的所有取值：
                    for ...
                        dp[状态1][状态2][...] = 择优(选择1，选择2...)
            对于股票买卖问题，总结的状态量一共有三个：天数/允许交易的最大次数/当前持有状态：dp[n][k][s]
'''


def maxProfit1(prices) -> int:
    """
    @brief      单次买卖股票
    @details    https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
    @param      prices -> list: 每天的股票价格
    @return     res -> int: 能获得的最大利润
    """
    # 构建dp table
    dp = [[0] * 2 for _ in range(len(prices) + 1)]
    dp[0][0] = 0
    # 不可能存在持有状态
    dp[0][1] = -float('inf')

    # TODO: 状态更新只于前一状态相关，可优化空间
    for i in range(1, len(prices) + 1):
        # i: 天数
        # dp[i][0]: 非持有状态,状态转移：保持原有不持有状态/卖掉之前的股票
        # dp[i][1]：持有状态,状态转移：保持原有持有状态/买入了股票，注意只买入一次，所以前置状态之前为0
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1])
        # dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1])
        dp[i][1] = max(dp[i - 1][1], 0 - prices[i - 1])

    # 收益最大的状态必然是非持有状态
    return dp[len(prices)][0]


def maxProfit2(prices) -> int:
    """
    @brief      多次买卖股票：但只能先买入再卖出
    @details    https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/
    @param      prices -> list: 每天的股票价格
    @return     res -> int: 能获得的最大利润
    """
    # 构建dp table
    dp = [[0] * 2 for _ in range(len(prices) + 1)]
    dp[0][0] = 0
    # 不可能存在持有状态
    dp[0][1] = -float('inf')

    for i in range(1, len(prices) + 1):
        # i: 天数
        # dp[i][0]: 非持有状态,状态转移：保持原有不持有状态/卖掉之前的股票
        # dp[i][1]：持有状态,状态转移：保持原有持有状态/买入了股票，与交易一次的区别在于买入的前置状态不为0
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1])
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1])

    # 收益最大的状态必然是非持有状态
    return dp[len(prices)][0]


def maxProfit3(prices) -> int:
    """
    @brief      多次买卖股票：但只能先买入再卖出
    @details    https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/
    @param      prices -> list: 每天的股票价格
    @return     res -> int: 能获得的最大利润
    """
    # 构建dp table
    dp = [[0] * 2 for _ in range(len(prices) + 1)]
    dp[0][0] = 0
    # 不可能存在持有状态
    dp[0][1] = -float('inf')

    for i in range(1, len(prices) + 1):
        # i: 天数
        # dp[i][0]: 非持有状态,状态转移：保持原有不持有状态/卖掉之前的股票
        # dp[i][1]：持有状态,状态转移：保持原有持有状态/买入了股票，与交易一次的区别在于买入的前置状态不为0
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1])
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1])

    # 收益最大的状态必然是非持有状态
    return dp[len(prices)][0]


if __name__ == '__main__':
    prices1_0 = [7, 1, 5, 3, 6, 4]
    prices1_1 = [7, 6, 4, 3, 1]
    prices2_1 = [1, 2, 3, 4, 5]
    res1_0 = maxProfit1(prices1_0)
    res1_1 = maxProfit1(prices1_1)
    res2_0 = maxProfit2(prices1_0)
    res2_1 = maxProfit2(prices2_1)
    print(res2_0)
    print(res2_1)

