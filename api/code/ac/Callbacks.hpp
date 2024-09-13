// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "CallbackResult.hpp"
#include <functional>
#include <string_view>

namespace ac {

/**
 * @brief Type alias for a simple callback function.
 *
 * @tparam R The type of result expected from the callback.
 *
 * This function is called when the asynchronous operation completes.
 */
template <typename R>
using BasicCb = std::function<void(R)>;

/**
 * @brief Type alias for a result callback function.
 *
 * @tparam R The type of the successful result.
 *
 * This function is called when the asynchronous operation completes,
 * either with a successful result or an error.
 */
template <typename R>
using ResultCb = BasicCb<CallbackResult<R>>;

/**
 * @brief Type alias for a progress callback function.
 *
 * This function is called to report progress during the asynchronous operation.
 * @param tag A string view representing a tag or category for the progress update.
 *            Since async tasks can be internally composed of multiple independent
 *            async tasks, tags allow for more granular progress reporting.
 * @param progress A float between 0 and 1
 */
using ProgressCb = std::function<void(std::string_view tag, float progress)>;

} // namespace ac
