// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <iostream>

#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/ChatFormat.hpp>
#include <ac/llama/Instance.hpp>

#include <jalog/Instance.hpp>
#include <jalog/sinks/ColorSink.hpp>

#include <vector>

int main() {
    jalog::Instance jl;
    jl.setup().add<jalog::sinks::ColorSink>();

    ac::llama::initLibrary();

    ac::llama::Model model("D:/mod/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q8_0.gguf", {});
    //ac::llama::Model model("D:/mod/gpt2/gguf/gpt2.Q6_K.gguf", {});
    //ac::llama::Model model("D:/mod/gpt2/gguf-gpt2-chatbot/gpt2-chatbot.Q8_0.gguf", {});

    ac::llama::Instance inst(model, {});

    inst.warmup();

    auto s = inst.newSession("The rain in Turkey", {});

    for (int i= 0; i < 8; ++i) {
        auto t = s.getToken();
        if (t == ac::llama::Token_Invalid) {
            break;
        }
        std::cout << model.vocab().tokenToString(t);
    }

    std::cout << std::endl;

    s.pushPrompt(" caused massive");

    for (int i = 0; i < 20; ++i) {
        auto t = s.getToken();
        if (t == ac::llama::Token_Invalid) {
            break;
        }
        std::cout << model.vocab().tokenToString(t);
    }

    std::cout << std::endl;

    return 0;
}
