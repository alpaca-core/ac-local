// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#pragma once
#include "export.h"
#include "PluginInfo.hpp"

#include <string_view>

#include <astl/safe_func.hpp>
#include <astl/ufunction.hpp>

namespace ac::local {
class ModelLoaderRegistry;
struct PluginInterface;

class AC_LOCAL_EXPORT PluginManager {
public:
    PluginManager(ModelLoaderRegistry& registry);
    ~PluginManager();

    PluginManager(const PluginManager&) = delete;
    PluginManager& operator=(const PluginManager&) = delete;

    static std::string_view pluginPathToName(std::string_view path);

    ModelLoaderRegistry& modelLoaderRegistry() const noexcept { return m_registry; }
    const std::vector<PluginInfo>& plugins() const noexcept { return m_plugins; }
    const std::vector<std::string>& pluginDirs() const noexcept { return m_pluginDirs; }

    void addPluginDir(std::string_view dir);
    void addPluginDirsFromEnvVar(std::string envVar);

    struct LoadPluginCb {
        template <typename Sig>
        using Func = astl::ufunction<Sig>;

        using PluginPathFilter = Func<bool(std::string_view name)>;
        PluginPathFilter pathFilter;
        using PluginNameFilter = Func<bool(std::string_view name)>;
        PluginNameFilter nameFilter;
        using PluginInterfaceFilter = Func<bool(const PluginInterface&)>;
        PluginInterfaceFilter interfaceFilter;
        using OnPluginLoaded = astl::safe_func<Func<void(const PluginInfo&)>>;
        OnPluginLoaded onPluginLoaded;
    };

    // load from path (including filename), ignores pluginDirs
    const PluginInfo* loadPlugin(const std::string& path, LoadPluginCb cb = {});

    // load all plugins from registered directories
    void loadPlugins(LoadPluginCb cb = {});

private:
    const PluginInfo* tryLoadPlugin(const std::string& path, LoadPluginCb& cb);

    ModelLoaderRegistry& m_registry;

    std::vector<std::string> m_pluginDirs;

    std::vector<PluginInfo> m_plugins;
};

} // namespace ac::local