// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/local.h>
#include <ac/local_dummy.h>
#include <ac/dict.h>

#include <jalogc.h>

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

#include "ac-test-data-dummy-models.h"

int main(void) {
    int ret = 0;
    jalogc_init((jalogc_init_params) {.add_default_sink = true});

    ac_local_model_factory* factory = ac_new_local_model_factory();
    ac_add_local_dummy_inference(factory);

    ac_local_model_desc_asset model_asset = {
        .path = AC_DUMMY_MODEL_SMALL,
        .tag = "x"
    };

    printf("Loading model...\n");
    ac_local_instance* instance = NULL;
    ac_local_model* model = ac_create_local_model(
        factory,
        "dummy", &model_asset, 1,
        ac_dict_arg_null(), NULL, NULL
    );
    if (!model) {
        ret = 1;
        goto cleanup;
    }

    printf("Creating instance...\n");
    instance = ac_create_local_instance(model, "general", ac_dict_arg_null());
    if (!instance) {
        ret = 1;
        goto cleanup;
    }

    printf("Generation:\n");
    ac_dict_root* params_root = ac_dict_new_root_from_json("{\"input\": [\"Xuxa\", \"sang:\"], \"splice\": false}", NULL);
    if (!params_root) {
        ret = 1;
        goto cleanup;
    }

    ac_dict_root* result_root = ac_dict_new_root();
    ac_dict_ref result = ac_run_local_op(ac_dict_make_ref(result_root), instance, "run",
        ac_dict_arg_take(ac_dict_make_ref(params_root)),
        NULL, NULL);

    ac_dict_free_root(params_root);
    if (!result) {
        ac_dict_free_root(result_root);
        ret = 1;
        goto cleanup;
    }

    char* dump = ac_dict_dump(result , 2);
    printf("%s\n", dump);
    free(dump);
    ac_dict_free_root(result_root);

cleanup:
    ac_free_local_instance(instance);
    ac_free_local_model(model);
    ac_free_local_model_factory(factory);
    return ret;
}
