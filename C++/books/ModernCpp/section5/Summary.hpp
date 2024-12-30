// Copyright (c) 2024

#ifndef _SUMMARY_HPP
#define _SUMMARY_HPP

#include "cpplang.hpp"
#include "SalesData.hpp"
#include "SpinLock.hpp"

END_NAMESPACE(ModernCpp)

class Summary final
{
public:
    using this_type = Summary;

public:
    using sales_type = SalesData;
    using lock_type = SpinLock;
    using lock_guard_type = SpinLockGuard;

    using string_type = std::string;
    using map_type = std::map<string_type, sales_type>;
    using minmax_sales_type = std::pair<string_type, string_type>;

private:
    map_type m_sales;
    minmax_sales_type m_minmax_sales;
    lock_type m_lock;


public:
    Summary() = default;
    ~Summary() = default;

    // 禁用拷贝构造和拷贝赋值
    Summary(const this_type&) = delete;
    Summary& operator=(const this_type&) = delete;

private:
    mutable lock_type m_lock; // 自旋锁：多线程用于保护共享数据，mutable：锁不影响类的状态
    map_type m_sales; // 存储销售数据

public:
    void add_sales(const sales_type& s)
    {
        lock_guard_type guard(m_lock);
        const auto& id = s.id();

        // not found
        auto it = m_sales.find(id);
        if (it == m_sales.end())
        {
            m_sales[id] = s;
            return;
        }
        else
        {
            it->second.inc_sold(s.sold());
            it->second.inc_revenue(s.revenue());
        }   
    }

    minmax_sales_type minmax_sales() const
    {
        lock_guard_type guard(m_lock);

        if (m_sales.empty())
        {
            return minmax_sales_type();
        }

        // algorithm: lambda函数，求最大最小值
        auto minmax = std::minmax_element(std::begin(m_sales), std::end(m_sales),
            [](const auto& a, const auto& b) {
                return a.second.sold() < b.second.sold();
            });

        auto min_pos = minmax.first;
        auto max_pos = minmax.second;

        return {min_pos->second.id(), max_pos->second.id()};
    }
};

END_NAMESPACE(ModernCpp)

#endif // _SUMMARY_HPP
