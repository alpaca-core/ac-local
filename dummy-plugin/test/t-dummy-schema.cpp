// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/local/Lib.hpp>
#include <ac/local/Model.hpp>
#include <ac/local/Instance.hpp>
#include <ac/local/ModelLoaderRegistry.hpp>

#include <ac/schema/Helpers.hpp>

#include <ac-test-util/JalogFixture.inl>

#include <doctest/doctest.h>

#include <dummy-schema.hpp>
#include <aclp-dummy-plib.hpp>
#include <ac-test-data-dummy-models.h>

struct LoadDummyFixture {
    LoadDummyFixture() {
        add_dummy_to_ac_local_global_registry();
    }
};

LoadDummyFixture loadDummyFixture;

TEST_CASE("dummy schema") {
    auto model = ac::local::Lib::modelLoaderRegistry().createModel({
        .type = "dummy",
        .name = "synthetic"
    }, {});

    REQUIRE(!!model);

    using Instance = ac::local::schema::Dummy::InstanceGeneral;
    auto instance = Model_createInstance<Instance>(*model, {.cutoff = 2});
    auto result = Instance_runOp<Instance::OpRun>(*instance, {.input = {"a", "b", "c"}});
    CHECK(result.result == "a one b two c one");
}