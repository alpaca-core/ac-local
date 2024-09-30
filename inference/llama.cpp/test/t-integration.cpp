// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/Instance.hpp>
#include <ac/llama/Session.hpp>

#include <doctest/doctest.h>

#include "ac-test-data-llama-dir.h"

struct GlobalFixture {
    GlobalFixture() {
        ac::llama::initLibrary();
    }
};

GlobalFixture globalFixture;

const char* Model_117m_q6_k = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";

TEST_CASE("vocab only") {
    ac::llama::Model model(Model_117m_q6_k, {}, { .vocabOnly = true });
    CHECK(!!model.lmodel());

    auto& params = model.params();
    CHECK(params.gpu);
    CHECK(params.vocabOnly);

    CHECK(model.trainCtxLength() == 0); // no weights - no training context
    CHECK_FALSE(model.shouldAddBosToken());
    CHECK_FALSE(model.hasEncoder());

    // vocab works
    auto& vocab = model.vocab();
    CHECK(vocab.tokenToString(443) == " le");
    CHECK(vocab.tokenize("hello world", true, true) == std::vector<ac::llama::Token>{31373, 995});
}

TEST_CASE("inference") {
    ac::llama::Model model(Model_117m_q6_k, {}, {});
    CHECK(!!model.lmodel());

    auto& params = model.params();
    CHECK(params.gpu);
    CHECK_FALSE(params.vocabOnly);

    CHECK(model.trainCtxLength() == 1024);
    CHECK_FALSE(model.shouldAddBosToken());
    CHECK_FALSE(model.hasEncoder());


    // general inference
    {
        ac::llama::Instance inst(model, {});
        inst.warmup(); // should be safe

        std::vector<ac::llama::Token> tokens;

        // choose a very, very suggestive prompt and hope that all architectures will agree
        auto s = inst.newSession({});
        tokens = model.vocab().tokenize("President George W.", true, true);
        s.setInitialPrompt(tokens);
        {
            auto t = s.getToken();
            REQUIRE(t != ac::llama::Token_Invalid);
            auto text = model.vocab().tokenToString(t);
            CHECK(text == " Bush");
        }

        // add more very suggestive stuff
        tokens = model.vocab().tokenize(" sent troops to Cleveland which was hit by torrential", false, false);
        s.pushPrompt(tokens);
        {
            auto t = s.getToken();
            REQUIRE(t != ac::llama::Token_Invalid);
            auto text = model.vocab().tokenToString(t);
            CHECK(text.starts_with(" rain")); // could be rains
        }
    }
}
