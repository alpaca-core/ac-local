// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "ModelLoaderRegistry.hpp"
#include "ModelLoader.hpp"
#include "Logging.hpp"
#include <astl/throw_stdex.hpp>
#include <astl/move.hpp>
#include <astl/qalgorithm.hpp>

namespace ac::local {

inline jalog::BasicStream& operator,(jalog::BasicStream& s, const std::vector<std::string>& vec) {
    s, "[";
    for (const auto& v : vec) {
        s, v, ", ";
    }
    s, "]";
    return s;
}

void ModelLoaderRegistry::addLoader(ModelLoader& loader, PluginInfo* plugin) {
    [[maybe_unused]] auto& info = loader.info();
    AC_LOCAL_LOG(Info, "Adding loader ", info.name,
        "\n       vendor: ", info.vendor,
        "\n  asset types: ", info.assetSchemaTypes,
        "\n  infer types: ", info.inferenceSchemaTypes,
        "\n         tags: ", info.tags
    );

    m_loaders.push_back({loader, plugin});
}

std::optional<ModelLoaderRegistry::LoaderData> ModelLoaderRegistry::findLoader(std::string_view schemaType) const noexcept {
    for (auto& data : m_loaders) {
        if (astl::pfind(data.loader.info().inferenceSchemaTypes, schemaType)) {
            return data;
        }
    }
    return std::nullopt;
}

ModelPtr ModelLoaderRegistry::createModel(ModelDesc desc, Dict params, ProgressCb cb) const {
    auto data = findLoader(desc.inferenceType);
    if (!data) {
        ac::throw_ex{} << "No loader found for schema type: " << desc.inferenceType;
    }
    return data->loader.loadModel(astl::move(desc), astl::move(params), astl::move(cb));
}

} // namespace ac::local
