// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/dict.h>
#include "s-unity.h"

void setUp(void) {}
void tearDown(void) {}

void parse(void) {
    const char* json = "{"
        "\"key\": \"value\","
        "\"seven\" : 7,"
        "\"obj\" : {"
            "\"nested\": \"nv\","
            "\"nested_bool\" : true"
        "},"
        "\"none\" : null,"
        "\"pi\" : 3.14159,"
        "\"ar\" : [1, -2, \"three\"]"
    "}";
    ac_dict_root* root = ac_dict_new_root_from_json(json, NULL);
    T_NOT_NULL(root);
    ac_dict_ref rr = ac_dict_make_ref(root);
    T_NOT_NULL(rr);

    CHECK(ac_dict_get_type(rr) == ac_dict_value_type_object);

    ac_dict_ref key = ac_dict_at_key(rr, "key");
    T_NOT_NULL(key);
    CHECK(ac_dict_get_type(key) == ac_dict_value_type_string);

    const char* key_str = ac_dict_get_string_value(key);
    T_EQ_STR("value", key_str);

    ac_dict_ref seven = ac_dict_at_key(rr, "seven");
    T_NOT_NULL(seven);
    CHECK(ac_dict_get_type(seven) == ac_dict_value_type_number_unsigned);
    T_EQ(7, ac_dict_get_int_value(seven));

    ac_dict_ref obj = ac_dict_at_key(rr, "obj");
    T_NOT_NULL(obj);
    CHECK(ac_dict_get_type(obj) == ac_dict_value_type_object);

    ac_dict_ref nested = ac_dict_at_key(obj, "nested");
    T_NOT_NULL(nested);
    CHECK(ac_dict_get_type(nested) == ac_dict_value_type_string);
    T_EQ_STR("nv", ac_dict_get_string_value(nested));

    ac_dict_ref nested_bool = ac_dict_at_key(obj, "nested_bool");
    T_NOT_NULL(nested_bool);
    CHECK(ac_dict_get_type(nested_bool) == ac_dict_value_type_bool);
    T_EQ(true, ac_dict_get_bool_value(nested_bool));

    ac_dict_ref none = ac_dict_at_key(rr, "none");
    T_NOT_NULL(none);
    CHECK(ac_dict_get_type(none) == ac_dict_value_type_null);

    ac_dict_ref pi = ac_dict_at_key(rr, "pi");
    T_NOT_NULL(pi);
    CHECK(ac_dict_get_type(pi) == ac_dict_value_type_number_double);
    T_CLOSE(1e-5, 3.14159, ac_dict_get_double_value(pi));

    ac_dict_ref ar = ac_dict_at_key(rr, "ar");
    T_NOT_NULL(ar);
    CHECK(ac_dict_get_type(ar) == ac_dict_value_type_array);

    ac_dict_ref ar0 = ac_dict_at_index(ar, 0);
    T_NOT_NULL(ar0);
    CHECK(ac_dict_get_type(ar0) == ac_dict_value_type_number_unsigned);
    T_EQ(1, ac_dict_get_int_value(ar0));

    ac_dict_ref ar1 = ac_dict_at_index(ar, 1);
    T_NOT_NULL(ar1);
    CHECK(ac_dict_get_type(ar1) == ac_dict_value_type_number_int);
    T_EQ(-2, ac_dict_get_int_value(ar1));

    ac_dict_ref ar2 = ac_dict_at_index(ar, 2);
    T_NOT_NULL(ar2);
    CHECK(ac_dict_get_type(ar2) == ac_dict_value_type_string);
    T_EQ_STR("three", ac_dict_get_string_value(ar2));

    ac_dict_free_root(root);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(parse);
    return UNITY_END();
}
