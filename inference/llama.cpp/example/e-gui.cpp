// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include <ac/llama/Init.hpp>
#include <ac/llama/Model.hpp>
#include <ac/llama/Instance.hpp>
#include <ac/llama/Session.hpp>
#include <ac/llama/AntipromptManager.hpp>

#include <imgui.h>
#include <imgui_stdlib.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_sdlrenderer2.h>
#include <SDL.h>

#include <jalog/Instance.hpp>
#include <jalog/Log.hpp>
#include <jalog/sinks/ColorSink.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <array>

#include "ac-test-data-llama-dir.h"

int sdlError(const char* msg) {
    std::cerr << msg << ": " << SDL_GetError() << "\n";
    return -1;
}

std::string_view get_filename(std::string_view path) {
    return path.substr(path.find_last_of('/') + 1);
}

void printModelLoadProgress(float progress) {
    const int barWidth = 50;
    static float currProgress = 0;

    auto delta = int(progress * barWidth) - int(currProgress * barWidth);

    if (delta) {
        printf("%s", std::string(delta, '=').c_str());
    }

    currProgress = progress;

    if (progress == 1.f) {
        std::cout << '\n';
        currProgress = 0.f;
    }
};

// unloadable model
class UModel {
public:
    UModel(std::string ggufPath) // intentionally implicit
        : m_ggufPath(std::move(ggufPath))
        , m_name(get_filename(m_ggufPath))
    {}

    const std::string& name() const { return m_name; }

    class State {
    public:
        State(const std::string& ggufPath, const ac::llama::Model::Params& modelParams)
            : m_model(ggufPath.c_str(), printModelLoadProgress, modelParams)
        {}

        class Instance {
        public:
            Instance(std::string name, ac::llama::Model& model, const ac::llama::Instance::InitParams& params)
                : m_name(std::move(name))
                , m_instance(model, params)
            {}


            class Session {
            public:
                Session(ac::llama::Instance& instance, std::string_view prompt, std::vector<std::string> antiprompts, ac::llama::Instance::SessionParams params)
                    : m_vocab(instance.model().vocab())
                    , m_params(std::move(params))
                    , m_text(std::move(prompt))
                    , m_session(instance.newSession(m_params))
                {
                    m_promptTokens = m_vocab.tokenize(m_text, true, true);
                    m_session.setInitialPrompt(m_promptTokens);
                    for (const auto& ap : antiprompts) {
                        m_antiprompt.addAntiprompt(ap);
                    }
                }

                const std::string& text() const { return m_text; }
                const ac::llama::Instance::SessionParams& params() const { return m_params; }

                void update() {
                    if (!m_numTokens) return;

                    auto token = m_session.getToken();
                    if (token == ac::llama::Token_Invalid) {
                        m_numTokens = 0;
                        return;
                    }

                    auto tokenStr = m_vocab.tokenToString(token);
                    m_text += tokenStr;

                    if (m_antiprompt.feedGeneratedText(tokenStr)) {
                        m_numTokens = 0;
                        return;
                    }

                    --m_numTokens;
                }

                void generate(uint32_t numTokens) {
                    m_numTokens = numTokens;
                }

                void pushPrompt(std::string_view prompt) {
                    m_text += "[";
                    m_text += prompt;
                    m_text += "]";
                    m_promptTokens = m_vocab.tokenize(prompt, false, true);
                    m_session.pushPrompt(m_promptTokens);
                }

            private:
                const ac::llama::Vocab& m_vocab;
                ac::llama::Instance::SessionParams m_params;
                std::vector<ac::llama::Token> m_promptTokens;
                std::string m_text;
                ac::llama::Session m_session;
                ac::llama::AntipromptManager m_antiprompt;
                uint32_t m_numTokens = 0;
            };

            const std::string& name() const { return m_name; }
            Session* session() { return m_session.get(); }

            void startSession(std::string_view prompt, std::vector<std::string> antiprompts, ac::llama::Instance::SessionParams params) {
                m_session.reset(new Session(m_instance, prompt, antiprompts, params));
            }

            void stopSession() {
                m_session.reset();
            }

        private:
            std::string m_name;
            ac::llama::Instance m_instance;
            std::unique_ptr<Session> m_session;
        };

        Instance* newInstance(const ac::llama::Instance::InitParams& params) {
            auto name = std::to_string(m_nextInstanceId++);
            m_instances.emplace_back(new Instance(name, m_model, params));
            return m_instances.back().get();
        }

        void dropInstance(Instance* i) {
            auto it = std::find_if(m_instances.begin(), m_instances.end(), [&](auto& ptr) { return ptr.get() == i; });
            if (it != m_instances.end()) {
                m_instances.erase(it);
            }
        }

        const std::vector<std::unique_ptr<Instance>>& instances() const { return m_instances; }
    private:
        ac::llama::Model m_model;

        int m_nextInstanceId = 0;
        std::vector<std::unique_ptr<Instance>> m_instances;
    };

    State* state() { return m_state.get(); }

    void unload() {
        m_state.reset();
        JALOG(Info, "unloaded ", m_name);
    }
    void load() {
        ac::llama::Model::Params modelParams;
        m_state.reset(new State(m_ggufPath, modelParams));
    }
private:
    std::string m_ggufPath;
    std::string m_name;

    std::unique_ptr<State> m_state;
};

int main(int, char**) {
    // setup logging
    jalog::Instance jl;
    jl.setup().add<jalog::sinks::ColorSink>();

    // setup llama
    ac::llama::initLibrary();

    // setup sdl
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0) {
        sdlError("Error: SDL_Init");
    }

    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");

    SDL_Window* window = SDL_CreateWindow(
        "Alpaca Core llama.cpp example",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 720,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window) {
        return sdlError("Error: SDL_CreateWindow");
    }
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr) {
        return sdlError("Error: SDL_CreateRenderer");
    }

    SDL_RendererInfo info;
    SDL_GetRendererInfo(renderer, &info);
    JALOG(Info, "SDL_Renderer: ", info.name);

    // setup imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    // setup backends
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);

    // app state

    auto models = std::to_array({
        UModel(AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf"),
        UModel(AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-f16.gguf")
    });
    UModel* selectedModel = models.data();
    UModel::State::Instance* selectedInstance = nullptr;

    ac::llama::Instance::InitParams newInstanceParams;
    ac::llama::Instance::SessionParams newSessionParams;

    std::string initialPrompt;
    std::string additionalPrompt;
    std::string antiprompt;
    uint32_t maxTokensToGenerate = 20;

    // main loop
    bool done = false;
    while (!done)
    {
        // frame setup
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                done = true;
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window))
                done = true;
        }

        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // app logic
        auto* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGui::Begin("#main", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize);

        ImGui::Text("FPS: %.2f (%.2gms)", io.Framerate, io.Framerate ? 1000.0f / io.Framerate : 0.0f);
        ImGui::Separator();

        ImGui::BeginTable("##main", 2, ImGuiTableFlags_Resizable);
        ImGui::TableNextColumn();

        ImGui::Text("Models");
        ImGui::BeginListBox("##models", {-1, 0});
        for (auto& m : models) {
            ImGui::PushID(&m);

            std::string name = m.name();
            if (m.state()) {
                name += " (0 sessions)";
            }
            else {
                name += " (unloaded)";
            }

            if (ImGui::Selectable(name.c_str(), selectedModel == &m)) {
                selectedModel = &m;
            }
            ImGui::PopID();
        }
        ImGui::EndListBox();
        UModel::State* modelState = nullptr;

        if (selectedModel) {
            if (selectedModel->state()) {
                if (ImGui::Button("Unload")) {
                    selectedModel->unload();
                }
            }
            else {
                if (ImGui::Button("Load")) {
                    selectedModel->load();
                }
            }

            modelState = selectedModel->state();
        }

        if (modelState) {
            ImGui::Separator();
            ImGui::Text("Instances");
            ImGui::BeginListBox("##instances", { -1, 0 });

            for (auto& i : modelState->instances()) {
                ImGui::PushID(i.get());

                std::string name = i->name();
                if (i->session()) {
                    name += " (active)";
                }
                else {
                    name += " (inactive)";
                }

                if (ImGui::Selectable(name.c_str(), selectedInstance == i.get())) {
                    selectedInstance = i.get();
                }
                ImGui::PopID();
            }
            ImGui::EndListBox();

            ImGui::BeginDisabled(!selectedInstance);
            if (ImGui::Button("Remove selected")) {
                modelState->dropInstance(selectedInstance);
            }
            ImGui::EndDisabled();

            ImGui::Separator();
            ImGui::Text("New instance");
            ImGui::InputScalar("ctxSize", ImGuiDataType_S32, &newInstanceParams.ctxSize);
            ImGui::InputScalar("batchSize", ImGuiDataType_S32, &newInstanceParams.batchSize);
            ImGui::InputScalar("ubatchSize", ImGuiDataType_S32, &newInstanceParams.ubatchSize);
            if (ImGui::Button("Create")) {
                selectedInstance = modelState->newInstance(newInstanceParams);
            }

            if (selectedInstance) {
                ImGui::TableNextColumn();
                if (auto session = selectedInstance->session()) {
                    session->update();
                    ImGui::Text("Session");
                    ImGui::TextWrapped("Params: gaFactor=%d, gaWidth=%d, infiniteContext=%d",
                        session->params().gaFactor, session->params().gaWidth, session->params().infiniteContext);

                    ImGui::Separator();
                    ImGui::TextWrapped("%s", session->text().c_str());
                    ImGui::Separator();

                    ImGui::InputTextMultiline("prompt", &additionalPrompt, {0, 50});
                    if (ImGui::Button("Push prompt")) {
                        session->pushPrompt(additionalPrompt);
                    }

                    ImGui::InputScalar("numTokens", ImGuiDataType_U32, &maxTokensToGenerate);
                    ImGui::SameLine();
                    if (ImGui::Button("Generate")) {
                        session->generate(maxTokensToGenerate);
                    }

                    ImGui::Separator();
                    if (ImGui::Button("Stop")) {
                        selectedInstance->stopSession();
                    }
                }
                else {
                    ImGui::Text("New session");
                    ImGui::Separator();
                    ImGui::InputScalar("gaFactor", ImGuiDataType_U32, &newSessionParams.gaFactor);
                    ImGui::InputScalar("gaWidth", ImGuiDataType_U32, &newSessionParams.gaWidth);
                    ImGui::Checkbox("infiniteContext", &newSessionParams.infiniteContext);
                    ImGui::InputTextMultiline("Initial prompt", &initialPrompt);
                    ImGui::InputTextMultiline("Anti prompt", &antiprompt);

                    std::vector<std::string> antiprompts;
                    auto splitAntiprompts = [&]() {
                        std::string tmp;
                        size_t processedChars = 0;
                        while(processedChars <= antiprompt.size()) {
                            if (antiprompt[processedChars] == '\n' || (processedChars == antiprompt.size())) {
                                antiprompts.push_back(tmp);
                                tmp.clear();
                                processedChars++;
                                continue;
                            }
                            tmp.push_back(antiprompt[processedChars++]);
                        }
                    };
                    splitAntiprompts();
                    if (ImGui::Button("Start")) {
                        selectedInstance->startSession(initialPrompt, antiprompts, newSessionParams);
                    }
                }
            }
        }

        ImGui::EndTable();
        ImGui::End();

        // frame render
        ImGui::Render();
        SDL_RenderSetScale(renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
        SDL_RenderClear(renderer);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);
    }

    // cleanup
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
