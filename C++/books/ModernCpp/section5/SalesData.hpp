// Copyright (c) 2024

#ifndef _SALES_DATA_HPP
#define _SALES_DATA_HPP

#include "cpplang.hpp"
#include <msgpack.hpp>

#if MSGPACK_VERSION_MAJOR < 2
# error "msgpack version must be >= 2.0.0"
#endif

BEGIN_NAMESPACE(ModernCpp)

// demo oop in C++
class SalesData final // 基础数据类：ID/销售册数/销售额
{
public:
    using this_type = SalesData;

public:
    using string_type = std::string;
    using string_view_type = std::string_view&;
    using uint_type = unsigned int;
    using currency_type = double;
    
    // 静态断言
    STATIC_ASSERT(sizeof(uint_type) >= 4);
    STATIC_ASSERT(sizeof(currency_type) >= 4);

public:
    SalesData(string_view_type id, uint_type s, currency_type r) noexcept
    : m_id(id), m_sold(s), m_revenue(r)
    {}

    SalesData(string_view_type id) noexcept // 委托构造函数
    : SalesData(id, 0, 0)
    {}

    SalesData() noexcept
    : SalesData("", 0, 0)
    {}

public:
    SalesData() = default;  // 显式default
    ~SalesData() = default;

    SalesData(const this_type&) = default;
    SalesData& operator=(const this_type&) = default;

    SalesData(this_type&& s) = default; // 显式转移构造
    SalesData& operator=(this_type&& s) = default;

private:
    string_type m_id = ""   ;
    uint_type m_sold = 0;
    currency_type m_revenue = 0;    

public:
    MSGPACK_DEFINE(m_id, m_sold, m_revenue);

    msgpack::sbuffer pack() const // 序列化
    {
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, *this);
        return buffer;
    }

    SalesData(const msgpack::sbuffer& buffer)
    {
        auto oh = msgpack::unpack(buffer.data(), buffer.size());
        oh.get().convert(*this);
    }

public:
    string_view_type id() const noexcept
    {
        return m_id;
    }

    uint_type sold() const noexcept
    {
        return m_sold;
    }
    currency_type revenue() const noexcept
    {
        return m_revenue;
    }

    CPP_DEPRECATED // 弃用标记
    currency_type average_price() const noexcept
    {
        return m_revenue / m_sold;
    }
};

END_NAMESPACE(ModernCpp)

#endif