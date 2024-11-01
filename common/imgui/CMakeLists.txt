# Copyright (c) Alpaca Core
# SPDX-License-Identifier: MIT
#
find_package(SDL2)
if(NOT SDL2_FOUND)
    message(STATUS "${CMAKE_PROJECT_NAME}: SDL was not found. ImGui examples will be unavailable")
    return()
endif()

CPMAddPackage(gh:ocornut/imgui@1.90.9)

icm_add_lib(imgui IMGUI
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlrenderer2.cpp
)
target_include_directories(imgui
    PUBLIC
        ${imgui_SOURCE_DIR}
    INTERFACE
        ${imgui_SOURCE_DIR}/backends
        ${imgui_SOURCE_DIR}/misc/cpp
)
target_link_libraries(imgui
    INTERFACE
        SDL2::SDL2main
    PUBLIC
        splat::splat # for SYMBOL_EXPORT
        SDL2::SDL2
)
target_compile_definitions(imgui PUBLIC IMGUI_USER_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/imgui_userconfig.h")
add_library(ac-dev::imgui ALIAS imgui)
