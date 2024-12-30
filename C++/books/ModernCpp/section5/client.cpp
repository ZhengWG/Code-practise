// Copyright (c) 2024

#include "cpplang.hpp"
#include "SalesData.hpp"
#include "Zmq.hpp"

// json.hpp shoulbe be in common/thirdparty/
#include "json.hpp"

USING_NAMESPACE(std);
USING_NAMESPACE(ModernCpp);

static auto debug_print = [](const auto& b) {
    using json_t = nlohmann::json;
    json_t j;

    j["id"] = b.id();
    j["sold"] = b.sold();
    j["revenue"] = b.revenue();

    cout << j.dump(2) << endl;
};

// sales data 序列化
static make_sales = [=](const auto& id, auto s, auto r) {
    return SalesData(id, s, r).pack();
};

// zmq send
static send_sales = [](const auto& addr, const auto& buf) {
    using zmq_ctx = ZmqContext<1>; // 1: 单线程
    auto sock = zmq_ctx::send_socket();

    sock.connect(addr);
    assert(sock.connected());
    
    auto len = sock.send(buf.data(), buf.size());
    assert(len == buf.size());

    cout << "send len = " << len << endl;
};

int main(){
    try{
        // 核心逻辑：
        // 1. 创建销售数据
        // 2. 发送销售数据

        count << "Hello client." << endl;

        auto buf = make_sales("001", 100, 1000);
        send_sales("tcp://localhost:5555", buf);

        this_thread::sleep_for(100ms);

        auto buf2 = make_sales("002", 200, 2000);
        send_sales("tcp://localhost:5555", buf);

    }
    catch(const exception& e){
        cerr << "Error: " << e.what() << endl;
    }
    catch(...){
        cerr << "Unknown error." << endl;
    }
}

