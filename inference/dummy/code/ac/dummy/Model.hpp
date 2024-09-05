// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include <string>
#include <vector>
#include <functional>

namespace ac::dummy {
using ModelLoadProgressCb = std::function<void(float)>;

class AC_DUMMY_EXPORT Model {
public:
    struct Params {
        std::string splice; // splice string before each data element
    };

    Model(const char* path, Params params);
    ~Model();

    const std::vector<std::string>& data() const noexcept { return m_data; }
private:
    const Params m_params;
    std::vector<std::string> m_data;
};

} // namespace ac::dummy