# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
ac_dep(test-data-whisper)

add_doctest_lib_test(integration ac-whisper
    SOURCES
        t-integration.cpp
    LIBRARIES
        ac-test-data::whisper
        ac-dev::audio
)
