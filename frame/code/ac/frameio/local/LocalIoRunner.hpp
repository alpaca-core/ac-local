// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "../../export.h"
#include "LocalIoCtx.hpp"
#include "../SessionHandlerPtr.hpp"

#include <astl/multi_thread_runner.hpp>
#include <cstdint>

namespace ac::frameio {

class BlockingIo;
struct LocalEndpoints;

// ideally this would be a nested type in LocalIoRunner, but then a clang bug is triggered:
// https://bugs.llvm.org/show_bug.cgi?id=36684
// to work around this, we have the type external
struct ChannelBufferSizes {
    size_t localToRemote = 10;
    size_t remoteToLocal = 10;
};

class AC_FRAME_EXPORT LocalIoRunner {
public:
    explicit LocalIoRunner(uint32_t numThreads = 2);
    LocalIoRunner(const LocalIoRunner&) = delete;
    LocalIoRunner& operator=(const LocalIoRunner&) = delete;
    ~LocalIoRunner();

    static LocalEndpoints getEndpoints(ChannelBufferSizes bufferSizes = {});

    void connect(SessionHandlerPtr remoteHandler, StreamEndpoint ep);
    StreamEndpoint connect(SessionHandlerPtr remoteHandler, ChannelBufferSizes bufferSizes = {});
    BlockingIo connectBlocking(SessionHandlerPtr remoteHandler, ChannelBufferSizes bufferSizes = {});
    void connect(SessionHandlerPtr local, SessionHandlerPtr remote, ChannelBufferSizes bufferSizes = {});

    void join(bool forceStop = false);

    LocalIoCtx& ctx() { return m_ctx; }

private:
    LocalIoCtx m_ctx;
    astl::multi_thread_runner<LocalIoCtx> m_threads;
};

} // namespace ac::frameio
