// Copyright (c) 2024

#ifndef _CONFIG_HPP
#define _CONFIG_HPP

#include "cpplang.hpp"

extern "C" { // 调用C库
#include <luajit.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include <LuaBridge/LuaBridge.h>

BEGIN_NAMESPACE(ModernCpp)

class Config final // 配置文件读取类： 使用Lua脚本语言
{
public:
    using vm_type = std::shared_ptr<lua_State>;  // Lua虚拟机类型
    using value_type = luabridge::LuaRef; // Lua值类型

    using string_type = std::string;
    using string_view_type = const std::string&;
    using regex_type = std::regex;
    using match_type = std::smatch;
public:
    Config() noexcept
    {
        assert(m_vm);
        luaL_openlibs(m_vm.get()); // 打开Lua标准库
    };
    ~Config() = default;
public:
    void load(string_view_type filename) const
    {
        auto status = luaL_dofile(m_vm.get(), filename.c_str());
        if (status != 0) {
            throw std::runtime_error("Failed to load config file");
        }
    }
    template<typename T> // 转换配置值的类型
    T get(string_view_type key) const // const常函数
    {
        if (!std::regex_match(key, m_what, m_regex)) { // 正则匹配
            throw std::runtime_error("Invalid key format");
        }

        auto w1 = m_what[1].str(); // 取出两个key
        auto w2 = m_what[2].str();
        auto value = luabridge::getGlobal(m_vm.get(), w1.c_str()); // 获取Lua表
        if (value.isNil()) {
            throw std::runtime_error("Key not found");
        }
        return luabridge::LuaRef_cast<T>(value[w2]);
    }


private:
    vm_type m_vm {luaL_newstate(), lua_close}; // 创建Lua虚拟机：初始化成员变量

    const regex_type m_regex {R"(^(\w+)\.(\w+)$)"};
    mutable match_type m_what; // 正则匹配的结果，不影响常量性，需要为mutable
};

END_NAMESPACE(ModernCpp)

#endif