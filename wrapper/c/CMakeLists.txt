# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
icm_auto_option(AC_C_WRAPPER "${CMAKE_PROJECT_NAME}: biuld c wrapper library" ${AC_WRAPPER_DEFAULT_VALUE})

if(NOT AC_C_WRAPPER)
    return()
endif()

if(AC_BUILD_TESTS)
    CPMAddPackage(gh:ThrowTheSwitch/Unity@2.6.0)
    set_target_properties(unity PROPERTIES FOLDER test)
    add_subdirectory(test-util)
endif()

add_subdirectory(cutil)
add_subdirectory(jalogc)

add_subdirectory(dict)

ac_add_local_subdir()
