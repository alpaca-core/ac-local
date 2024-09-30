// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/llama/Session.hpp>
#include <doctest/doctest.h>
#include <vector>
#include <deque>

using namespace ac::llama;
using TokenVec = std::vector<Token>;

// the session is a coroutine wrapper
// we can test it in isolation with a fake instance (no actual AI inference going on)
// in this case we create a coroutine which:
// * receives an initial prompt
// * for every subsequent prompt, it yields the sum of prompt tokens and the initial prompt tokens (zipped, wrapped)

Session TestSession() {
    auto initialPrompt = co_await Session::Prompt{};

    if (initialPrompt.empty()) {
        throw std::runtime_error("empty");
    }

    const TokenVec initial(initialPrompt.begin(), initialPrompt.end());
    auto iiter = initial.begin();

    initialPrompt = {};

    bool yieldedInvalid = false;

    co_await Session::StartGeneration{};

    std::deque<Token> current;
    current.push_back(Token(initial.size()));

    while (true) {
        auto prompt = co_await Session::Prompt{};

        if (prompt.empty()) {
            if (current.empty() && yieldedInvalid) {
                current.push_back(1); // ok one more
            }
        }
        else {
            // rewrite current
            current.assign(prompt.begin(), prompt.end());
        }

        yieldedInvalid = false;

        if (current.empty()) {
            co_yield Token_Invalid;
            yieldedInvalid = true;
        }
        else {
            if (current.front() == 0xBADF00D) {
                throw std::runtime_error("test exception");
            }
            co_yield current.front() + *iiter; // zip
            current.pop_front();
            ++iiter;
            if (iiter == initial.end()) {
                // wrap
                iiter = initial.begin();
            }
        }
    }
}

TEST_CASE("session default") {
    Session s;
    CHECK(!s);
}

TEST_CASE("session empty initial") {
    auto s = TestSession();
    CHECK_THROWS_WITH(s.setInitialPrompt({}), "empty");
}

TEST_CASE("session no push") {
    Session s = TestSession();

    {
        TokenVec initialPrompt = {1, 2, 3};
        s.setInitialPrompt(initialPrompt);
    }

    REQUIRE(s);

    CHECK(s.getToken() == 4);
    CHECK(s.getToken() == Token_Invalid);
    CHECK(s.getToken() == 3);
    CHECK(s.getToken() == Token_Invalid);
    CHECK(s.getToken() == 4);
    CHECK(s.getToken() == Token_Invalid);
    CHECK(s.getToken() == 2);
    CHECK(s.getToken() == Token_Invalid);
}

TEST_CASE("session push") {
    Session s = TestSession();

    {
        TokenVec initialPrompt = {1, 2, 3};
        s.setInitialPrompt(initialPrompt);
    }

    REQUIRE(s);

    TokenVec prompt = {4, 5};
    s.pushPrompt(prompt);
    CHECK(s.getToken() == 5);
    CHECK(s.getToken() == 7);
    CHECK(s.getToken() == Token_Invalid);

    prompt = {10, 20, 30, 40};

    s.pushPrompt(prompt);
    CHECK(s.getToken() == 13);

    prompt.clear(); // safe to destroy after first getToken

    CHECK(s.getToken() == 21);

    SUBCASE("no replace") {
        CHECK(s.getToken() == 32);
        CHECK(s.getToken() == 43);
        CHECK(s.getToken() == Token_Invalid);
    }
    SUBCASE("replace") {
        prompt = {100, 200};
        s.pushPrompt(prompt);
        CHECK(s.getToken() == 102);
        CHECK(s.getToken() == 203);
        CHECK(s.getToken() == Token_Invalid);
    }

    CHECK(s.getToken() == 2);
    CHECK(s.getToken() == Token_Invalid);
}

TEST_CASE("exceptions") {
    Session s = TestSession();

    SUBCASE("empty initial") {
        CHECK_THROWS_WITH(s.setInitialPrompt({}), "empty");
    }
    SUBCASE("bad prompt") {
        {
            TokenVec initialPrompt = { 1, 2, 3 };
            s.setInitialPrompt(initialPrompt);
        }

        REQUIRE(s);

        TokenVec prompt = { 4, 0xBADF00D };
        s.pushPrompt(prompt);
        CHECK(s.getToken() == 5);
        CHECK_THROWS_WITH(s.getToken(), "test exception");
    }

    // everything is invalid from now on
    CHECK(s.getToken() == Token_Invalid);
    CHECK(s.getToken() == Token_Invalid);
}
