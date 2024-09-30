// Copyright (c) Alpaca Core
// SPDX-License-Identifier: MIT
//
#include "Instance.hpp"
#include "Model.hpp"
#include "Logging.hpp"
#include "Session.hpp"
#include <llama.h>
#include <astl/throw_ex.hpp>
#include <astl/iile.h>
#include <astl/move.hpp>
#include <itlib/sentry.hpp>
#include <cassert>
#include <span>

namespace ac::llama {

namespace {
llama_context_params llamaFromInstanceInitParams(Model& model, const Instance::InitParams& params) {
    llama_context_params llamaParams = llama_context_default_params();
    llamaParams.n_ctx = params.ctxSize;
    llamaParams.n_batch = params.batchSize;
    llamaParams.n_ubatch = params.ubatchSize;
    llamaParams.flash_attn = model.params().gpu;
    return llamaParams;
}
} // namespace

Instance::Instance(Model& model, InitParams params)
    : m_model(model)
    , m_sampler(model, {})
    , m_lctx(llama_new_context_with_model(model.lmodel(), llamaFromInstanceInitParams(model, params)), llama_free)
{
    if (!m_lctx) {
        throw_ex{} << "Failed to create llama context";
    }
    assert(model.lmodel() == llama_get_model(m_lctx.get()));

    const auto ctxLen = llama_n_ctx(m_lctx.get());
    const auto ctxTrain = model.trainCtxLength();
    if (ctxLen > ctxTrain) {
        LLAMA_LOG(Warning, "Instance requested context length ", ctxLen, " is greater than the model's training context length ", ctxTrain);
    }
}

Instance::~Instance() = default;

void Instance::warmup() {
    LLAMA_LOG(Info, "Running warmup");

    auto lctx = m_lctx.get();
    auto model = m_model.lmodel();

    std::vector<llama_token> tmp;
    llama_token bos = llama_token_bos(model);
    llama_token eos = llama_token_eos(model);
    // some models (e.g. T5) don't have a BOS token
    if (bos != LLAMA_TOKEN_NULL) {
        tmp.push_back(bos);
    }
    if (eos != LLAMA_TOKEN_NULL) {
        tmp.push_back(eos);
    }
    if (tmp.empty()) {
        tmp.push_back(0);
    }

    const auto ntokens = int32_t(tmp.size());

    if (llama_model_has_encoder(model)) {
        llama_encode(lctx, llama_batch_get_one(tmp.data(), ntokens, 0, 0));
        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = bos;
        }
        tmp.clear();
        tmp.push_back(decoder_start_token_id);
    }
    llama_decode(lctx, llama_batch_get_one(tmp.data(), ntokens, 0, 0));
    llama_kv_cache_clear(lctx);
    llama_synchronize(lctx);
    llama_perf_context_reset(lctx);
}

Session Instance::newSession(const SessionParams params) {
    // see comments in Session::promiseType::initial_suspend on why this silly await is necessary
    auto initialPrompt = co_await Session::Prompt{};

    if (m_hasActiveSession) {
        throw_ex{} << "Instance already has an active session";
    }
    m_hasActiveSession = true;
    itlib::sentry closeSessionSentry([this] { m_hasActiveSession = false; });

    auto lctx = m_lctx.get();
    auto& vocab = m_model.vocab();

    llama_kv_cache_clear(lctx);
    llama_synchronize(lctx);
    llama_perf_context_reset(lctx);
    m_sampler.reset();
    m_sampler.perfReset();

    Token initialToken; // used to reset the initial prompt to a single token

    if (initialPrompt.empty()) {
        // Should not run without any tokens
        initialToken = llama_token_bos(m_model.lmodel());
        initialPrompt = {&initialToken, 1};
    }

    if (initialPrompt.empty()) {
        throw_ex{} << "Empty initial prompt";
    }

    const auto ctxLen = llama_n_ctx(lctx);
    const auto maxTokens = ctxLen - 4; // ref #16
    if (initialPrompt.size() > maxTokens) {
        throw_ex{} << "Initial prompt too long. Got " << initialPrompt.size() << " tokens, max: " << ctxLen - 4;
    }

    const auto numKeep = int32_t(initialPrompt.size()); // number of tokens to keep in the context in case we overflow

    if (params.gaFactor != 1) {
        const uint32_t gaFactor = params.gaFactor;
        const uint32_t gaWidth = params.gaWidth;
        if (gaWidth % gaFactor != 0) {
            throw_ex{} << "Group-attention width " << gaWidth << " must be a multiple of group-attention factor " << gaFactor;
        }
        LLAMA_LOG(Info, "self-extend: train = ", m_model.trainCtxLength(), ", gaFactor = ", gaFactor, ", gaWidth = ", gaWidth);
    }

    if (m_model.hasEncoder()) {
        // well llama_encode does not touch the tokens, but llama_batch needs them to be non-const
        // (mostly for stupid C reasons)
        // so... we have to do something evil here
        auto nonConstTokens = const_cast<Token*>(initialPrompt.data());
        auto batch = llama_batch_get_one(nonConstTokens, int32_t(initialPrompt.size()), 0, 0);
        auto res = llama_encode(lctx, batch);
        if (res != 0) {
            throw_ex{} << "Failed to encode input";
        }
        initialToken = vocab.decoderStartToken();
        initialPrompt = {&initialToken, 1};
    }

    // group attention state
    uint32_t gaIndex = 0; // number of grouped KV tokens (only used if params.gaFactor > 1)
    uint32_t numPast = 0; // number of tokens in the context (that's prompts + generated)

    enum class Source {
        InitialPrompt,
        InteractivePrompt,
        Generated
    };

    auto doDecode = [&](std::span<const Token> tokens, Source src) {
        // first try to expand the context if needed
        const auto gaFactor = params.gaFactor;

        // Ensure the input doesn't exceed the context size by truncating embd if necessary.
        if (tokens.size() > maxTokens) {
            const auto skipped = tokens.size() - maxTokens;
            tokens = tokens.first(maxTokens);
            LLAMA_LOG(Warning, "Input too long. Skipping ", skipped, " tokens");
        }

        bool haveFullContextMitigation = false;
        if (gaFactor == 1) {
            // infinite text generation via context shifting
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via numPast)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            const auto num = numPast + tokens.size();
            if (num >= ctxLen) {
                if (!params.infiniteContext) {
                    throw_ex{} << "context limit of " << ctxLen << " reached";
                }

                const auto numLeft = numPast - numKeep;
                const int numDiscard = numLeft / 2; // somewhat arbitrary

                LLAMA_LOG(Debug, "Context is full. Swapping: past = ", numPast, ", numLeft: ", numLeft,
                    ", ctxLen: ", ctxLen, ", numKeep: ", numKeep, ", numDiscard: ", numDiscard);

                llama_kv_cache_seq_rm(lctx, 0, numKeep, numKeep + numDiscard);
                llama_kv_cache_seq_add(lctx, 0, numKeep + numDiscard, numPast, -numDiscard);

                numPast -= numDiscard;
                haveFullContextMitigation = true;
            }
        }
        else {
            const uint32_t gaWidth = params.gaWidth;

            while (numPast >= gaIndex + gaWidth) {
                // context extension via Self-Extend
                const int ib = (gaFactor * gaIndex) / gaWidth;
                const int bd = (gaWidth / gaFactor) * (gaFactor - 1);
                const int dd = (gaWidth / gaFactor) - ib * bd - gaWidth;

                LLAMA_LOG(Debug, "Group attention shift: ib = ", ib, ", bd = ", bd, ", dd = ", dd);

                llama_kv_cache_seq_add(lctx, 0, gaIndex, numPast, ib * bd);
                llama_kv_cache_seq_div(lctx, 0, gaIndex + ib * bd, gaIndex + ib * bd + gaWidth, gaFactor);
                llama_kv_cache_seq_add(lctx, 0, gaIndex + ib * bd + gaWidth, numPast + ib * bd, dd);

                numPast -= bd;

                gaIndex += gaWidth / gaFactor;
                haveFullContextMitigation = true;
            }
        }

        if (haveFullContextMitigation) {
            LLAMA_LOG(Info, "Context full mitigation performed: past = ", numPast, ", tokens = ", tokens.size());
        }

        // add to sampler
        for (auto t : tokens) {
            // only apply grammar for generated content
            m_sampler.accept(t, src == Source::Generated);
        }

        // decode
        const auto batchSize = llama_n_batch(lctx);

        // decode with batches of batchSize
        while (!tokens.empty()) {
            auto batchTokens = tokens.size() > batchSize ? tokens.first(batchSize) : tokens;
            tokens = tokens.subspan(batchTokens.size());

            // well llama_decode does not touch the tokens, but llama_batch needs them to be non-const
            // (mostly for stupid C reasons)
            // so... we have to do something evil here
            auto nonConstTokens = const_cast<Token*>(batchTokens.data());
            auto batch = llama_batch_get_one(nonConstTokens, int(batchTokens.size()), numPast, 0);
            if (llama_decode(lctx, batch) != 0) {
                throw_ex{} << "Failed to decode tokens";
            }
            numPast += uint32_t(batchTokens.size());
        }
    };

    doDecode(initialPrompt, Source::InitialPrompt);
    initialPrompt = {}; // clear as after the co_await below the memory may be invalid

    co_await Session::StartGeneration{}; // suspend pre generation

    while (true) {
        auto prompt = co_await Session::Prompt{};
        if (!prompt.empty()) {

            // reset sampling and don't allow previous inputs to affect the generation
            m_sampler.reset();

            doDecode(prompt, Source::InteractivePrompt);
        }

        auto token = m_sampler.sample(lctx);
        if (vocab.isEog(token)) {
            co_yield Token_Invalid;
            // don't decode eog tokens in case the the interaction is continued
        }
        else {
            // first yield, then decode, thus we don't decode if the session is aborted
            co_yield token;
            doDecode({&token, 1}, Source::Generated);
        }
    }
}

} // namespace ac::llama
