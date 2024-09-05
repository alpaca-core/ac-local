// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/api.h>
#include <ac/dict.h>
#include <ac/local_provider.h>
#include <ac/local_dummy.h>
#include <ac/asset_source_local_dir.h>

#include <jalogc.h>

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdatomic.h>

#include "ac-test-data-dummy-models.h"

#if !defined(UNUSED)
#define UNUSED(x) (void)(x)
#endif

typedef struct app_state {
    ac_model* model;
    ac_instance* instance;
    atomic_bool cur_step_done;
} app_state;

void init_app_state(app_state* state) {
    state->model = NULL;
    state->instance = NULL;
    atomic_init(&state->cur_step_done, false);
}

void free_state(app_state* state) {
    ac_free_model(state->model);
    ac_free_instance(state->instance);
}

void wait_for_step(app_state* state) {
    // just spin lock
    // it's the easiest way and effective resource management is not a priority in this demo
    while (!atomic_load(&state->cur_step_done)) {
        // spin
    }
    atomic_store(&state->cur_step_done, false); // prepare next step
}

void on_progress(float progress, void* user_data) {
    UNUSED(user_data);
    printf("Progress: %f\n", progress);
}

void on_model_result(ac_model* m, const char* error, void* user_data) {
    app_state* state = (app_state*)user_data;
    assert(state->model == NULL);
    state->model = m;
    if (error) {
        printf("Model load error: %s\n", error);
    }
    atomic_store(&state->cur_step_done, true);
}

void on_instance_result(ac_instance* i, const char* error, void* user_data) {
    app_state* state = (app_state*)user_data;
    assert(state->instance == NULL);
    state->instance = i;
    if (error) {
        printf("Instance creation error: %s\n", error);
    }
    atomic_store(&state->cur_step_done, true);
}

void on_run_op_result(const char* error, void* user_data) {
    app_state* state = (app_state*)user_data;
    if (error) {
        printf("Run op error: %s\n", error);
    }
    atomic_store(&state->cur_step_done, true);
}

void on_op_stream(ac_dict_ref dict, void* user_data) {
    app_state* state = (app_state*)user_data;
    ac_dict_ref result = ac_dict_at_key(dict, "result");
    if (result) {
        printf("%s\n", ac_dict_get_string_value(result));
    }
    else {
        char* dump = ac_dict_dump(dict, 2);
        printf("Error bad dict:\n%s\n", dump);
    }
    atomic_store(&state->cur_step_done, true);
}


int main(void) {
    jalogc_init((jalogc_init_params) { .add_default_sink = true });

    ac_api_provider* local_provider = ac_new_local_api_provider();
    app_state state;

    int ret = 0;
    init_app_state(&state);

    ac_add_local_dummy_inference(local_provider);
    ac_add_asset_source(local_provider, ac_new_asset_source_local_dir(AC_TEST_DATA_DUMMY_DIR), 0);

    {
        ac_model_info_asset asset = { .id = AC_DUMMY_MODEL_SMALL_ASSET };
        ac_add_model(local_provider, "dummy-small", "dummy", &asset, 1);
    }

    printf("Loading model...\n");
    ac_create_model(
        local_provider,
        "dummy-small",
        NULL,
        on_model_result,
        on_progress,
        &state
    );
    wait_for_step(&state);
    if (!state.model) {
        ret = 1;
        goto cleanup;
    }

    printf("Creating instance...\n");
    ac_create_instance(
        state.model,
        "general",
        NULL,
        on_instance_result,
        on_progress,
        &state
    );
    wait_for_step(&state);
    if (!state.instance) {
        ret = 1;
        goto cleanup;
    }

    printf("Generation: ");
    ac_run_op(
        state.instance,
        "run",
        ac_dict_new_root_from_json("{\"input\": [\"Xuxa\", \"sang:\"], \"splice\": false}", NULL),
        on_run_op_result,
        on_op_stream,
        &state
    );
    wait_for_step(&state);

cleanup:
    free_state(&state);
    ac_free_api_provider(local_provider);
    return ret;
}