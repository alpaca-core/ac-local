// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Provider.hpp"
#include "Model.hpp"
#include "Instance.hpp"

// export vtables for classes which only have that

namespace ac::api {
Provider::~Provider() = default;
Model::~Model() = default;
Instance::~Instance() = default;
}