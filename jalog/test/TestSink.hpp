// Copyright (c) Borislav Stanimirov
// SPDX-License-Identifier: MIT
//
#include <doctest/doctest.h>

#include <ac/jalog/Sink.hpp>
#include <ac/jalog/Entry.hpp>

#include "TestTime.hpp"

#include <string>
#include <cstdint>
#include <vector>
#include <algorithm>

class TestSink final : public ac::jalog::Sink
{
public:
    virtual void record(const ac::jalog::Entry& entry) override
    {
        entries.emplace_back(entry);
    }

    struct EntryCopy
    {
        EntryCopy() = default;
        EntryCopy(const ac::jalog::Entry& e)
            : scope(e.scope)
            , level(e.level)
            , timestamp(tonano(e.timestamp))
            , text(e.text)
        {}
        ac::jalog::ScopeDesc scope;
        ac::jalog::Level level;
        uint64_t timestamp;
        std::string text;
    };

    std::vector<EntryCopy> entries;

    void checkSameEntries(const TestSink& other) const
    {
        auto& es0 = entries;
        auto& es = other.entries;
        CHECK(es0.size() == es.size());
        for (size_t ei = 0; ei < es0.size(); ++ei)
        {
            auto& e0 = es0[ei];
            auto& e = es[ei];
            CHECK(&e0 != &e);

            CHECK(e0.scope.label() == e.scope.label());
            CHECK(e0.scope.id() == e.scope.id());
            CHECK(e0.scope.userData == e.scope.userData);
            CHECK(e0.level == e.level);
            CHECK(e0.timestamp == e.timestamp);
            CHECK(e0.text == e.text);
        }
    }

    static void checkSortedEntries(const std::vector<EntryCopy>& es)
    {
        CHECK(std::is_sorted(es.begin(), es.end(), [](auto& a, auto& b) {
            return a.timestamp < b.timestamp;
        }));
    }
};
