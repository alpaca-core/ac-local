# Alpaca Core

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20) [![Build](https://github.com/alpaca-core/alpaca-core/actions/workflows/build.yml/badge.svg)](https://github.com/alpaca-core/alpaca-core/actions/workflows/build.yml)

AI Inference SDK

## Bindings, Wrappers, and Integrations

* C: API wrapper [in repo](wrapper/c)
* Java
    * Pure java wrapper [in repo](wrapper/java)
    * [Android SDK with this repo as a submodule](https://github.com/alpaca-core/ac-android)
* Swift


## Demos

## Build

Use CMake. Some useful presets are provided in the repo.

### CMake Options

* `AC_BUILD_LOCAL` - build the local inference api and integrations of inference libraries
* `AC_INFERENCE_LIBS` - `all` or `none` or semicolon separated list. Which inference libraries to build. The supported ones are:
    * `dummy`: a dummy inference library for testing and experimenting
    * `llama.cpp`: wrapping our fork of [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
    * `whisper`: wrapping our fork of [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp)
* Advanced and `OFF` by default if the project is a subdir:
    * `AC_BUILD_TESTS` - build tests. This means all tests for all active components. No finer-grain options are available yet
    * `AC_BUILD_EXAMPLES` - build examples. This means all examples for all active components. No finer-grain options are available yet
    * `AC_BUILD_POC` - build proof of concept, sandbox, and experimental projects

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This software is distributed under the MIT Software License. See accompanying file LICENSE or copy [here](https://opensource.org/licenses/MIT).

Copyright &copy; 2024 [Alpaca Core, Inc](https://github.com/alpaca-core)
