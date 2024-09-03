# Alpaca Core

AI Inference SDK

## Build

Use CMake. Some useful presets are provided in the repo.

### Options

* `AC_BUILD_API` - build the public API. If `OFF`, only inference libraries will be available)
    * `AC_BUILD_LOCAL_PROVIDER` - build the local inference provider and api integrations of inference libraries
* `AC_INFERENCE_LIBS` - `all` or `none` or semicolon separated list. Which inference libraries to build
* Advanced and `OFF` by default if the project is a subdir:
    * `AC_BUILD_TESTS` - build tests. This means all tests for all active components. No finer-grain options are available yet
    * `AC_BUILD_EXAMPLES` - build examples. This means all examples for all active components. No finer-grain options are available yet
    * `AC_BUILD_POC` - build proof of concept, sandbox, and experimental projects

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This software is distributed under the MIT Software License. See accompanying file LICENSE or copy [here](https://opensource.org/licenses/MIT).

Copyright &copy; 2024 Alpaca Core
