// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include "Token.hpp"
#include <vector>
#include <string_view>

namespace ac::llama {

class Model;

struct AC_LLAMA_EXPORT Vocab {
public:
    Vocab(const Model& model);
    ~Vocab();

    std::vector<Token> tokenize(std::string_view text, bool addSpecial, bool parseSpecial) const;

    Token decoderStartToken() const noexcept; // fallback to bos if not available

private:
    const Model& m_model;
};

} // namespace ac::llama
