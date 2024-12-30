// Copyright (c) 2024

#include "cpplang.hpp"
#include "Summary.hpp"
#include "Zmq.hpp"
#include "Config.hpp"

// you should put json.hpp in common/thirdparty/
#include "json.hpp"

#include <cstdio>
#include <cpr/cpr.h>

USING_NAMESPACE(std);
USING_NAMESPACE(ModernCpp);

static debug_print = [](const auto& b) {
    using json_t = nlohmann::json;
    json_t j;
    
    j["id"] = b.id();
    j["sold"] = b.sold();
    j["revenue"] = b.revenue();

    cout << j.dump(2) << endl;
};

int main(){
    try{
        // 核心逻辑：
        // 1. 加载配置
        // 2. 创建数据存储类Summary
        // 3. 创建主循环：recv_cycle + log_cycle
        // 4. 启动主循环
        // 5. 等待主循环结束
        cout << "Hello server." << endl;
        Config cfg;
        cfg.load("./conf.lua");

        Summary summary;
        std::atomic_int count(0); // 原子计数器

        auto recv_cycle = [&](){
            using zmq_ctx = ZmqContext<1>;
            
            // zmq recv
            auto sock = zmq_ctx::recv_sock(); // 接收socket

            sock.bind(conf.get<string>("config.zmq_ipc_addr")); // 绑定ZMQ接收端口
            assert(sock.connected());

            for(;;){ // 服务器无限循环
                // xxx : shared_ptr/unique_ptr
                auto msg_ptr = std::make_shared<zmq_message_type>();

                sock.recv(msg_ptr.get()); // ZMQ：堵塞接收数据                
                ++count;

                // async process msg
                std::thread([&sum, &count]() // 启动线层做反序列化存储，主动捕获，防止堵塞接收数据
                {
                    SalesData b;
                    auto obj = msgpack::unpack(msg_ptr->data<char>(), msg_ptr->size()).get();
                    obj.convert(b);
                    sum.add_sales(b); // 存储数据
                }
                ).detach(); // 分离线程，异步运行
            }
        };

        auto log_cycle = [&](){
            auto http_addr = conf.get<string>("config.http_addr");
            auto time_interval = conf.get<int>("config.time_interval");
            for(;;){
                this_thread::sleep_for(time_interval * 1s);
                
                auto info = sum.minmax_sales();
                cout << "log_cycle get info" << endl;

                using json_t = nlohmann::json;
                json_t j;
                j["count"] = static_cast<int>(count);
                j["minmax"] = sum.minmax_sales(); // {info.first, info.second}

                auto res = cpr::Post(
                    cpr::Url{http_addr}, 
                    cpr::Body{j.dump()},
                    cpr::Timeout{200ms}); // 设置超时
                if(res.status_code != 200){
                    cerr << "log_cycle post failed, status: " << res.status_code << endl;
                }
            }
        };

        // launch log_cycle
        auto fu1 = std::async(std::launch::async, log_cycle);

        // launch recv_cycle
        auto fu2 = std::async(std::launch::async, recv_cycle);

        fu2.wait();
    }
    catch(const exception& e){
        cerr << "Error: " << e.what() << endl;
    }
    catch(...){
        cerr << "Unknown error." << endl;
    }
}