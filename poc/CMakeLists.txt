# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
add_subdirectory(sandbox)
add_subdirectory(xec-coro)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/personal-sandbox/CMakeLists.txt")
    add_subdirectory(personal-sandbox)
endif()


