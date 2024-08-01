// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/LocalProvider.hpp>
#include <ac/LocalLlama.hpp>

#include <ac/Model.hpp>
#include <ac/Instance.hpp>

#include <jalog/Instance.hpp>
#include <jalog/sinks/ColorSink.hpp>

#include <iostream>
#include <optional>
#include <latch>

#include "ac-test-data-llama-dir.h"

int main() {
    jalog::Instance jl;
    jl.setup().add<jalog::sinks::ColorSink>();

    ac::LocalProvider provider;
    ac::addLocalLlamaInference(provider);

    std::optional<std::latch> latch;

    ac::CallbackResult<ac::ModelPtr> modelResult;
    latch.emplace(1);
    std::string gguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";
    provider.createModel({{"type", "llama.cpp"}, {"gguf", gguf}}, {
        [&](ac::CallbackResult<ac::ModelPtr> result) {
            modelResult = std::move(result);
            latch->count_down();
        },
        [](float f) {
            std::cout << "model load progress: " << f << "\n";
        }
    });

    latch->wait();

    if (modelResult.has_error()) {
        std::cout << "model load error: " << modelResult.error().text << "\n";
        return 1;
    }
    auto model = std::move(modelResult.value());

    ac::CallbackResult<ac::InstancePtr> instanceResult;
    latch.emplace(1);
    model->createInstance("general", {}, {
        [&](ac::CallbackResult<ac::InstancePtr> result) {
            instanceResult = std::move(result);
            latch->count_down();
        },
        {} // empty progress callback
    });

    latch->wait();
    if (instanceResult.has_error()) {
        std::cout << "instance create error: " << instanceResult.error().text << "\n";
        return 1;
    }
    auto instance = std::move(instanceResult.value());

    std::string prompt = "The first person to";
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Generation: ";

    std::string opError;
    latch.emplace(1);
    instance->runOp("run", {{"prompt", prompt}, {"max_tokens", 20}}, {
        [&](ac::CallbackResult<void> result) {
            if (result.has_error()) {
                opError = std::move(result.error().text);
                return;
            }
            latch->count_down();
        },
        [](ac::Dict result) {
            std::cout << result.at("result").get<std::string_view>();
        }
    });

    latch->wait();

    std::cout << "\n";

    if (!opError.empty()) {
        std::cout << "run error: " << opError << "\n";
        return 1;
    }

    return 0;
}