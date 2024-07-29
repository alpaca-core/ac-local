// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"

namespace ac {

class AC_API_EXPORT Instance {
public:
    virtual ~Instance();

    // ops are queued and executed in order
    // the inner state is modified by ops and subsequent ones may depend on the results of previous ones
    virtual void runOp(std::string_view op, Dict params, Callback<void, Dict> cb) = 0;

    // wait for all ops to finish
    virtual void synchronize() = 0;

    // Note that some callbacks may still be called after this function.
    // Either wait for pending callbacks or call synchronize after this to guarantee that no more callbacks will be called.
    virtual void initiateAbort(Callback<void> cb) = 0;
};

} // namespace ac