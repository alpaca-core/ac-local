# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
add_subdirectory(code)

if(AC_LOCAL_BUILD_TESTS OR AC_LOCAL_BUILD_EXAMPLES)
    add_subdirectory(test-models)
endif()

ac_local_add_test_subdir()

if(AC_LOCAL_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
