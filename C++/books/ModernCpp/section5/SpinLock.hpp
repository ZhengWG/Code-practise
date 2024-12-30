// Copyright (c) 2024

#ifndef _SPIN_LOCK_HPP
#define _SPIN_LOCK_HPP

#include "cpplang.hpp"

BEGIN_NAMESPACE(ModernCpp)

// atomic spinlock with TAS
class SpinLock final // 自旋锁类：保护多线程下的共享资源，防止竞争
{
public:
    using this_type = SpinLock; // 类型别名
    using atomic_type = std::atomic_flag;

public:
    SpinLock() = default; // 默认构造函数
    ~SpinLock() = default;

    SpinLock(const this_type&) = delete; // 禁止拷贝
    SpinLock& operator=(const this_type&) = delete;

public:
    void lock() noexcept // 自旋锁定，不抛出异常
    {
        for(;;) { // 无限循环
            // 原子变量操作：TAS（Compare-and-Swap）
            if (!m_lock.test_and_set()) {
                return; // 成功锁定，退出循环
            }
            std::this_thread::yield(); // 解除自旋锁定
        }
    }
    void unlock() noexcept
    {
        m_lock.clear(); // 解锁
    }

    bool try_lock() noexcept{
        return !m_lock.test_and_set();
    }

private:
    atomic_type m_lock {false};
};

// RAII for lock
class SpinLockGuard final // 自旋锁RAII类，自动解锁
{
public:
    using this_type = SpinLockGuard; // 类型别名
    using spin_lock_type = SpinLock;
public:
    SpinLockGuard(spin_lock_type& lock) noexcept : m_lock(lock) { m_lock.lock(); }
    ~SpinLockGuard() noexcept { m_lock.unlock(); }

public:
    SpinLockGuard(const this_type&) = delete;
    SpinLockGuard& operator=(const this_type&) = delete;

private:
    spin_lock_type& m_lock;
};
#endif
