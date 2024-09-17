// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//

// a dummy synchronous provider used in tests only

#include <ac/Model.hpp>
#include <ac/Instance.hpp>
#include <ac/ApiCUtil.hpp>

namespace {
class DummyInstance final : public ac::Instance {
    void runOp(std::string_view op, ac::Dict params, OpCallback cb) override {
        std::string_view tag = "stream";

        cb.progressCb(tag, 1.f);

        if (op == "insta") {
            cb.streamCb({{"insta", "success"}});
            cb.completionCb({});
            return;
        }

        cb.progressCb(tag, 2.f);
        cb.streamCb({{"some", 42}});

        if (op == "error") {
            cb.completionCb(itlib::unexpected(ac::Error{params.at("error").get<std::string>()}));
        }
        else {
            cb.streamCb({{"more", 1024}});
            cb.completionCb({});
        }
    }

    void synchronize() override {} // not really testable

    void initiateAbort(ac::BasicCb<void> cb) override {
        cb();
    }
};

class DummyModel final : public ac::Model {
    void createInstance(std::string_view type, ac::Dict params, ac::ResultCb<ac::InstancePtr> cb) override {
        if (type == "error") {
            cb(itlib::unexpected(ac::Error{params.at("error").get<std::string>()}));
            return;
        }
        cb(ac::InstancePtr{std::make_shared<DummyInstance>()});
    }
};

} // anonymous namespace

extern "C" ac_model* create_dummy_model() {
    return ac::cutil::ac_model_create(std::make_shared<DummyModel>());
}
