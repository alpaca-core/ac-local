# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
add_subdirectory(whisper.cpp)
add_subdirectory(code)
ac_add_local_subdir()
ac_add_test_subdir()
ac_add_example_subdir()
