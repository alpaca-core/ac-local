// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace ac::llama {
class AC_LLAMA_EXPORT IncrementalStringFinder {
public:
    IncrementalStringFinder(std::string searchStr = "");

    // incremental search for `m_str` in `text`
    bool feedText(std::string_view text);

private:
    std::string m_searchStr;
    uint16_t m_currentPos;
};
}
