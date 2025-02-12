// Copyright (c) Borislav Stanimirov
// SPDX-License-Identifier: MIT
//

// copied from https://github.com/iboB/splat/tree/master/splat

// Intentionally leave no include guard
// Thus one can define symbols before including

// Shared library interface
#if !defined(SYMBOL_IMPORT)
#   if defined(_WIN32)
#       define SYMBOL_EXPORT __declspec(dllexport)
#       define SYMBOL_IMPORT __declspec(dllimport)
#   else
#       define SYMBOL_EXPORT __attribute__((__visibility__("default")))
#       define SYMBOL_IMPORT
#   endif
#endif
