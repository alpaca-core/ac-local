// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//

// trivial example of using alpaca-core's llama inference

// llama
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/Instance.hpp>
#include <ac/llama/Session.hpp>

// logging
#include <jalog/Instance.hpp>
#include <jalog/sinks/ColorSink.hpp>

// model source directory
#include "ac-test-data-llama-dir.h"

#include <iostream>
#include <string>

int main() try {
    jalog::Instance jl;
    jl.setup().add<jalog::sinks::ColorSink>();

    // initialize the library
    ac::llama::initLibrary();

    // load model
    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";
    ac::llama::Model::Params modelParams;
    auto modelLoadProgressCallback = [](float progress) {
        const int barWidth = 50;
        static float currProgress = 0;
        auto delta = int(progress * barWidth) - int(currProgress * barWidth);
        for (int i = 0; i < delta; i++) {
            std::cout.put('=');
        }
        currProgress = progress;
        if (progress == 1.f) {
            std::cout << '\n';
        }
        return true;
    };
    ac::llama::Model model(modelGguf.c_str(), modelLoadProgressCallback, modelParams);

    // create inference instance
    ac::llama::Instance instance(model, {});

    std::string prompt = "The first person to";
    std::cout << "Prompt: " << prompt << "\n";

    // start session
    auto session = instance.newSession({});
    session.setInitialPrompt(model.vocab().tokenize(prompt, true, true));

    // generate and print 100 tokens
    for (int i = 0; i < 100; ++i) {
        auto token = session.getToken();
        if (token == ac::llama::Token_Invalid) {
            // no more tokens
            break;
        }
        std::cout << model.vocab().tokenToString(token);
    }
    std::cout << '\n';

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
