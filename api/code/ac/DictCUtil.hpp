// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "Dict.hpp"
#include "dict_ref.h"
#include "dict_root.h"

/**
 * @file DictCUtil.hpp
 * @brief Utility functions for converting between C and C++ dictionary representations in the Alpaca Core library.
 *
 * This file provides functions to bridge the gap between the C API's dictionary representations
 * (ac_dict_root and ac_dict_ref) and the C++ Dict type. These utilities are essential for
 * implementing the C API functions that interact with the underlying C++ dictionary implementation.
 */

// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include "Dict.hpp"
#include "dict_ref.h"
#include "dict_root.h"

namespace ac::cutil {

/**
 * @brief Consumes a C dictionary root and returns the contained C++ Dict object.
 *
 * This function takes ownership of the ac_dict_root, extracts the Dict object,
 * and then frees the root structure. If the input is null, it returns an empty Dict.
 *
 * @param d Pointer to the dictionary root to consume.
 * @return Dict A Dict object created from the consumed dictionary root.
 */
AC_API_EXPORT Dict Dict_from_dict_root_consume(ac_dict_root* d);

/**
 * @brief Converts a C dictionary reference to a C++ Dict object reference.
 *
 * This function provides access to the underlying Dict object referenced by an ac_dict_ref.
 * It's used internally to implement C API functions that need to operate on the C++ Dict.
 *
 * @param d The dictionary reference to convert.
 * @return Dict& A reference to the converted Dict object.
 */
AC_API_EXPORT Dict& Dict_from_dict_ref(ac_dict_ref d);

/**
 * @brief Converts a C++ Dict object to a C dictionary reference.
 *
 * This function creates an ac_dict_ref that points to the given Dict object.
 * It's used internally to implement C API functions that need to return references
 * to Dict objects.
 *
 * @param d The Dict object to convert.
 * @return ac_dict_ref The resulting dictionary reference.
 */
AC_API_EXPORT ac_dict_ref Dict_to_dict_ref(Dict& d);

/**
 * @brief Parses a JSON string into a Dict object.
 *
 * This function uses the nlohmann::json parser to convert a JSON string into a Dict object.
 * It's used internally to implement JSON parsing functionality in the C API.
 *
 * @param json The JSON string to parse.
 * @param jsonEnd Optional pointer to the end of the JSON string. If nullptr, the entire string is parsed.
 * @return Dict The resulting Dict object parsed from the JSON string.
 */
AC_API_EXPORT Dict Dict_parse(const char* json, const char* jsonEnd = nullptr);

} // namespace ac::cutil
