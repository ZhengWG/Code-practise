// Copyright (c) 2024

#ifndef _ZMQ_HPP
#define _ZMQ_HPP

#include "cpplang.hpp"
#include <zmq.hpp>

BEGIN_NAMESPACE(ModernCpp)

using zmq_context_type = zmq::context_t;
using zmq_socket_type = zmq::socket_t;
using zmq_message_type = zmq::message_t;

template<int thread_num = 1> // 使用整数模板参数来指定线程数
class ZmqContext final
{
# if 0 // 临时禁用代码
public:
    using zmq_context_type = zmq::context_t;
    using zmq_socket_type = zmq::socket_t;
    using zmq_message_type = zmq::message_t;
# endif
public:
    ZmqContext() = default;
    ~ZmqContext() = default;

public:
   // 静态成员函数代替静态成员变量：C++要求静态成员变量必须在cpp文件里定义实现，头文件里只是声明
   static zmq_socket_type recv_sock(int hwm = 1000, int linger = 10)
   {
    zmq_socket_type sock(context(), ZMQ_PULL);
    sock.setsockopt(ZMQ_RCVHWM, hwm);
    sock.setsockopt(ZMQ_LINGER, linger); // wait for $linger seconds
    return sock;
   }

   static zmq_socket_type send_sock(int hwm = 1000, int linger = 10)
   {
    zmq_socket_type sock(context(), ZMQ_PUSH);
    sock.setsockopt(ZMQ_SNDHWM, hwm);
    sock.setsockopt(ZMQ_LINGER, linger); // wait for $linger seconds
    return sock;
   }
};

END_NAMESPACE(ModernCpp)

#endif