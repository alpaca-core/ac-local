# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#

include(./cmake/AddSwift.cmake)

add_library(AlpacaCoreSwift STATIC
    Expected.hpp
    Dict.hpp
    Dict.cpp
    Model.hpp
    Model.cpp
    Instance.hpp
    Instance.cpp
    AlpacaCore.hpp
    AlpacaCore.cpp
    SwiftApi.swift
    DictHelper.swift
    ModelDesc.swift
)

# If we don't clear compiler options -Wall is passed which is not recognized by Swift compiler
set_target_properties(AlpacaCoreSwift PROPERTIES COMPILE_OPTIONS "")

target_compile_options(AlpacaCoreSwift
    PUBLIC
        $<$<COMPILE_LANGUAGE:Swift>:-cxx-interoperability-mode=default -Xcc -std=c++20 -emit-symbol-graph -emit-symbol-graph-dir "${CMAKE_BINARY_DIR}/../../doc/swift/symbol-graph">
)

target_include_directories(AlpacaCoreSwift PUBLIC .)

target_link_libraries(AlpacaCoreSwift
    PUBLIC
        ac::dict
        ac::local
    PRIVATE
        ac-local-dummy
        ac-local-whisper
        ac-local-llama
        jalog::jalog
        jalog::sinklib
)

if(CMAKE_GENERATOR STREQUAL Xcode)
    set_target_properties(AlpacaCoreSwift PROPERTIES
        XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET}
    )
endif()

_swift_generate_cxx_header(AlpacaCoreSwift
    AlpacaCore-Swift.h
    SWIFT_EXPOSABLE_FILES ModelDesc.swift
)
