// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include <ac/api_provider.h>

#if defined(__cplusplus)
extern "C" {
#endif

AC_LOCAL_WHISPER_EXPORT void ac_add_local_whisper_inference(ac_api_provider* local_provider);

#if defined(__cplusplus)
}
#endif