// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "Token.hpp"
#include <span>
#include <utility>
#include <exception>
#include <coroutine>

namespace ac::llama {

class Session {
public:
    struct Prompt {}; // sentinel for co_await-int prompts
    struct StartGeneration {}; // sentinel for co_await-ing the start of generation

    class promise_type {
        Token m_value = Token_Invalid;

        std::span<const Token> m_pendingPrompt;

        std::exception_ptr m_exception;
    public:
        Session get_return_object() noexcept {
            return Session{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        Token value() const noexcept { return m_value; }

        // ideally we would use std::suspend_never here and have an eager coroutine
        // HOWEVER clang is stupid (or maybe pedantic is the right word)
        // There is a defect in the standard tracked here https://github.com/cplusplus/CWG/issues/575
        // MSVC and gcc do something reasonable if an exception is thrown in an eager coroutine before the first
        // actual suspend
        // clang however simply leaks the coroutine state and doesn't even call the destructors of local variables
        // so, until the issue is resolved and compilers are updated (likely years from now) we must have a lazy
        // coroutine here and use the setInitialPrompt shenanigans to work around it
        std::suspend_always initial_suspend() noexcept { return {}; }

        // keep the coroutine alive to make the last yield value available
        std::suspend_always final_suspend() noexcept { return {}; }

        std::suspend_always yield_value(Token value) noexcept {
            m_value = value;
            return {};
        }

        void return_void() noexcept {
            m_value = Token_Invalid;
        }
        void unhandled_exception() noexcept {
            // why store the exception here instead of simply rethrowing?
            // because clang is stupid, that's why
            // clang's ridiculous handling of coroutines causes local coroutine variables to be destroyed twice if we
            // just rethrow here
            m_exception = std::current_exception();
        }

        struct Awaiter {
            promise_type& self;
            bool await_ready() noexcept { return true; }
            std::span<const Token> await_resume() noexcept {
                // clear pending after returning it
                return std::exchange(self.m_pendingPrompt, std::span<const Token>{});
            }
            void await_suspend(std::coroutine_handle<>) noexcept {}
        };

        void setPrompt(std::span<const Token> prompt) {
            m_pendingPrompt = prompt;
        }

        Awaiter await_transform(Prompt) noexcept { return Awaiter{*this}; }

        std::suspend_always await_transform(StartGeneration) noexcept { return {}; }

        void rethrowIfException() {
            if (m_exception) {
                std::rethrow_exception(m_exception);
            }
        }
    };

    using Handle = std::coroutine_handle<promise_type>;

    Session() = default;
    Session(Handle handle) : m_handle(handle) {}

    Session(Session&& other) noexcept : m_handle(std::exchange(other.m_handle, nullptr)) {}
    Session& operator=(Session&& other) noexcept {
        if (this == &other) return *this;
        if (m_handle) {
            m_handle.destroy();
        }
        m_handle = std::exchange(other.m_handle, nullptr);
        return *this;
    }

    ~Session() {
        if (m_handle) {
            m_handle.destroy();
        }
    }

    void reset() {
        if (m_handle) {
            m_handle.destroy();
        }
        m_handle = nullptr;
    }

    explicit operator bool() const noexcept { return !!m_handle; }

    // the provided span must remain valid until the next call to getToken or pushPrompt with another span
    void pushPrompt(std::span<const Token> prompt) {
        m_handle.promise().setPrompt(prompt);
    }

    void setInitialPrompt(std::span<const Token> prompt) {
        m_handle.promise().setPrompt(prompt);
        m_handle.resume();
        m_handle.promise().rethrowIfException();
    }

    Token getToken() {
        if (m_handle.done()) return Token_Invalid;
        m_handle.resume();
        m_handle.promise().rethrowIfException();
        return std::move(m_handle.promise().value());
    }

private:
    Handle m_handle;
};

} // namespace ac::llama
