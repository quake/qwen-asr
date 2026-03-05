/*
 * qwen_asr_decoder.c - Qwen3 LLM decoder
 *
 * Architecture (per layer):
 *   RMSNorm -> QKV (no bias) -> per-head Q/K RMSNorm -> NeoX RoPE
 *   -> Causal GQA attention -> Output proj -> residual
 *   RMSNorm -> SwiGLU MLP (gate/up/down, no bias) -> residual
 *
 * Features: Q/K per-head RMSNorm, NeoX split-half RoPE, GQA 2:1,
 * tied embeddings (tok_embeddings == lm_head).
 */

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static double dec_get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <sys/time.h>
static double dec_get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

static float *load_f32(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

int qwen_decoder_load(qwen_decoder_t *dec, multi_safetensors_t *ms,
                       const qwen_config_t *cfg) {
    char name[512];

    /* Token embeddings (large, bf16 mmap direct) */
    dec->tok_embeddings_bf16 = load_bf16_direct(ms,
        "thinker.model.embed_tokens.weight");
    if (!dec->tok_embeddings_bf16) return -1;

    /* Transformer layers */
    for (int i = 0; i < cfg->dec_layers; i++) {
        qwen_dec_layer_t *l = &dec->layers[i];

        /* Attention weights (bf16, no bias) */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.q_proj.weight", i);
        l->wq_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.k_proj.weight", i);
        l->wk_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.v_proj.weight", i);
        l->wv_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.o_proj.weight", i);
        l->wo_weight_bf16 = load_bf16_direct(ms, name);

        /* Per-head Q/K RMSNorm weights */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.q_norm.weight", i);
        l->q_norm_weight = load_f32(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.k_norm.weight", i);
        l->k_norm_weight = load_f32(ms, name);

        /* RMSNorm weights */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.input_layernorm.weight", i);
        l->input_norm = load_f32(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.post_attention_layernorm.weight", i);
        l->post_attn_norm = load_f32(ms, name);

        /* SwiGLU MLP weights (bf16, no bias) */
        snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.gate_proj.weight", i);
        l->gate_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.up_proj.weight", i);
        l->up_weight_bf16 = load_bf16_direct(ms, name);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.down_proj.weight", i);
        l->down_weight_bf16 = load_bf16_direct(ms, name);

        if (!l->wq_weight_bf16 || !l->wk_weight_bf16 ||
            !l->wv_weight_bf16 || !l->wo_weight_bf16 ||
            !l->gate_weight_bf16 || !l->up_weight_bf16 || !l->down_weight_bf16) {
            fprintf(stderr, "decoder: failed to load layer %d\n", i);
            return -1;
        }

        /* Fuse gate+up weights: interleave rows [gate_row0, up_row0, gate_row1, up_row1, ...] */
        {
            int inter = cfg->dec_intermediate;
            int hidden = cfg->dec_hidden;
            size_t row_bytes = (size_t)hidden * sizeof(uint16_t);
            l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * (size_t)inter * row_bytes);
            for (int r = 0; r < inter; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r) * hidden,
                       l->gate_weight_bf16 + (size_t)r * hidden, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2 * r + 1) * hidden,
                       l->up_weight_bf16 + (size_t)r * hidden, row_bytes);
            }
        }

    }

    /* Final RMSNorm */
    dec->norm = load_f32(ms, "thinker.model.norm.weight");
    if (!dec->norm) return -1;

    return 0;
}

/* ========================================================================
 * KV Cache Management
 * ======================================================================== */

static int kv_cache_init(qwen_ctx_t *ctx, int max_seq) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    size_t cache_size = (size_t)ctx->config.dec_layers * max_seq * kv_dim * sizeof(float);
    ctx->kv_cache_k = (float *)calloc(1, cache_size);
    ctx->kv_cache_v = (float *)calloc(1, cache_size);
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;
    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return -1;
    return 0;
}

static int kv_cache_grow(qwen_ctx_t *ctx, int required) {
    if (required <= ctx->kv_cache_max) return 0;

    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    int new_max = ctx->kv_cache_max;
    while (new_max < required) new_max *= 2;

    size_t new_stride = (size_t)new_max * kv_dim;
    size_t old_stride = (size_t)ctx->kv_cache_max * kv_dim;
    size_t total = (size_t)ctx->config.dec_layers * new_stride * sizeof(float);

    float *new_k = (float *)calloc(1, total);
    float *new_v = (float *)calloc(1, total);
    if (!new_k || !new_v) { free(new_k); free(new_v); return -1; }

    size_t copy = (size_t)ctx->kv_cache_len * kv_dim * sizeof(float);
    for (int l = 0; l < ctx->config.dec_layers; l++) {
        memcpy(new_k + l * new_stride, ctx->kv_cache_k + l * old_stride, copy);
        memcpy(new_v + l * new_stride, ctx->kv_cache_v + l * old_stride, copy);
    }

    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    ctx->kv_cache_k = new_k;
    ctx->kv_cache_v = new_v;
    ctx->kv_cache_max = new_max;
    return 0;
}

static float *kv_cache_k_at(qwen_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static float *kv_cache_v_at(qwen_ctx_t *ctx, int layer, int pos) {
    int kv_dim = ctx->config.dec_kv_heads * ctx->config.dec_head_dim;
    return ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static int ensure_prefill_buffers(qwen_ctx_t *ctx, int seq_len) {
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;

    if (seq_len <= ctx->pref_seq_cap) return 0;

    int new_cap = ctx->pref_seq_cap > 0 ? ctx->pref_seq_cap : 64;
    while (new_cap < seq_len) new_cap *= 2;

#define REALLOC_PREF(ptr, count) do {                                          \
    void *tmp__ = realloc((ptr), (size_t)(count) * sizeof(float));             \
    if (!tmp__) return -1;                                                      \
    (ptr) = (float *)tmp__;                                                     \
} while (0)

    REALLOC_PREF(ctx->pref_x, new_cap * dim);
    REALLOC_PREF(ctx->pref_x_norm, new_cap * dim);
    REALLOC_PREF(ctx->pref_q, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_k, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_v, new_cap * kv_dim);
    REALLOC_PREF(ctx->pref_attn_out, new_cap * q_dim);
    REALLOC_PREF(ctx->pref_proj_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_ffn_out, new_cap * dim);
    REALLOC_PREF(ctx->pref_gate, new_cap * intermediate);
    REALLOC_PREF(ctx->pref_gate_up, new_cap * 2 * intermediate);

#undef REALLOC_PREF

    ctx->pref_seq_cap = new_cap;
    return 0;
}

static int ensure_rope_inv_freq(qwen_ctx_t *ctx, int head_dim, float theta) {
    int half = head_dim / 2;
    if (ctx->rope_inv_freq && ctx->rope_inv_freq_half == half) return 0;

    float *inv = (float *)realloc(ctx->rope_inv_freq, (size_t)half * sizeof(float));
    if (!inv) return -1;
    ctx->rope_inv_freq = inv;

    for (int d = 0; d < half; d++) {
        ctx->rope_inv_freq[d] = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    }
    ctx->rope_inv_freq_half = half;
    return 0;
}

static int ensure_rope_cache(qwen_ctx_t *ctx, int required_pos, int head_dim, float theta) {
    if (required_pos <= ctx->rope_cache_cap) return 0;
    if (ensure_rope_inv_freq(ctx, head_dim, theta) != 0) return -1;

    int new_cap = ctx->rope_cache_cap > 0 ? ctx->rope_cache_cap : 1024;
    while (new_cap < required_pos) new_cap *= 2;

    size_t n = (size_t)new_cap * head_dim;
    float *new_cos = (float *)realloc(ctx->rope_cache_cos, n * sizeof(float));
    if (!new_cos) return -1;
    ctx->rope_cache_cos = new_cos;

    float *new_sin = (float *)realloc(ctx->rope_cache_sin, n * sizeof(float));
    if (!new_sin) return -1;
    ctx->rope_cache_sin = new_sin;

    int half = head_dim / 2;
    for (int pos = ctx->rope_cache_cap; pos < new_cap; pos++) {
        float p = (float)pos;
        float *cos_row = ctx->rope_cache_cos + (size_t)pos * head_dim;
        float *sin_row = ctx->rope_cache_sin + (size_t)pos * head_dim;
        for (int d = 0; d < half; d++) {
            float angle = p * ctx->rope_inv_freq[d];
            float c = cosf(angle);
            float s = sinf(angle);
            cos_row[d] = c;
            cos_row[half + d] = c;
            sin_row[d] = s;
            sin_row[half + d] = s;
        }
    }

    ctx->rope_cache_cap = new_cap;
    return 0;
}

/* ========================================================================
 * Decoder Prefill (Multiple Tokens)
 * ======================================================================== */

void qwen_decoder_prefill(qwen_ctx_t *ctx, const float *input_embeds, int seq_len) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Ensure KV cache */
    if (!ctx->kv_cache_k) {
        if (kv_cache_init(ctx, seq_len + 1024) != 0) return;
    } else if (ctx->kv_cache_len + seq_len > ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, ctx->kv_cache_len + seq_len + 1024) != 0) return;
    }

    if (ensure_prefill_buffers(ctx, seq_len) != 0) return;

    float *x = ctx->pref_x;
    float *x_norm = ctx->pref_x_norm;
    float *q = ctx->pref_q;
    float *k = ctx->pref_k;
    float *v = ctx->pref_v;
    float *attn_out = ctx->pref_attn_out;
    float *proj_out = ctx->pref_proj_out;
    float *ffn_out = ctx->pref_ffn_out;
    float *gate = ctx->pref_gate;
    float *gate_up = ctx->pref_gate_up;

    memcpy(x, input_embeds, (size_t)seq_len * dim * sizeof(float));

    int start_pos = ctx->kv_cache_len;
    if (ensure_rope_cache(ctx, start_pos + seq_len, head_dim, theta) != 0) return;
    const float *rope_cos = ctx->rope_cache_cos + (size_t)start_pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)start_pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    /* Profiling accumulators */
    double prof_rms_norm_ms = 0, prof_qkv_proj_ms = 0, prof_qk_norm_rope_ms = 0;
    double prof_kv_store_ms = 0, prof_attn_ms = 0, prof_out_proj_ms = 0;
    double prof_ffn_gate_up_ms = 0, prof_ffn_swiglu_ms = 0, prof_ffn_down_ms = 0;
    double prof_residual_ms = 0;
    double t0;

    if (qwen_verbose >= 2) qwen_linear_nobias_bf16_prof_reset();

    int n_layers = cfg->dec_layers;
    if (ctx->dec_layers_limit > 0 && ctx->dec_layers_limit < n_layers)
        n_layers = ctx->dec_layers_limit;

    for (int layer = 0; layer < n_layers; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        /* Input RMSNorm */
        t0 = dec_get_time_ms();
        qwen_rms_norm(x_norm, x, l->input_norm, seq_len, dim, eps);
        prof_rms_norm_ms += dec_get_time_ms() - t0;

        /* QKV projections (no bias) — includes bf16->f32 conversion */
        t0 = dec_get_time_ms();
        qwen_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, seq_len, dim, q_dim);
        qwen_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, seq_len, dim, kv_dim);
        qwen_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, seq_len, dim, kv_dim);
        prof_qkv_proj_ms += dec_get_time_ms() - t0;

        /* Per-head Q/K RMSNorm + NeoX RoPE */
        t0 = dec_get_time_ms();
        qwen_rms_norm_per_head(q, l->q_norm_weight, seq_len, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, seq_len, n_kv_heads, head_dim, eps);
        qwen_apply_rope_neox(q, rope_cos, rope_sin, seq_len, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, seq_len, n_kv_heads, head_dim);
        prof_qk_norm_rope_ms += dec_get_time_ms() - t0;

        /* Store K, V in cache */
        t0 = dec_get_time_ms();
        for (int s = 0; s < seq_len; s++) {
            memcpy(kv_cache_k_at(ctx, layer, start_pos + s),
                   k + s * kv_dim, kv_dim * sizeof(float));
            memcpy(kv_cache_v_at(ctx, layer, start_pos + s),
                   v + s * kv_dim, kv_dim * sizeof(float));
        }
        prof_kv_store_ms += dec_get_time_ms() - t0;

        /* Causal attention */
        t0 = dec_get_time_ms();
        int total_seq = start_pos + seq_len;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);
        qwen_causal_attention(attn_out, q, full_k, full_v,
                               seq_len, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, start_pos);
        prof_attn_ms += dec_get_time_ms() - t0;

        /* Output projection + residual */
        t0 = dec_get_time_ms();
        qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16,
                                 seq_len, q_dim, dim);
        prof_out_proj_ms += dec_get_time_ms() - t0;

        t0 = dec_get_time_ms();
        qwen_add_inplace(x, proj_out, seq_len * dim);
        prof_residual_ms += dec_get_time_ms() - t0;

        /* Post-attention RMSNorm */
        t0 = dec_get_time_ms();
        qwen_rms_norm(x_norm, x, l->post_attn_norm, seq_len, dim, eps);
        prof_rms_norm_ms += dec_get_time_ms() - t0;

        /* SwiGLU MLP — gate_up projection (includes bf16->f32) */
        t0 = dec_get_time_ms();
        qwen_linear_nobias_bf16(gate_up, x_norm, l->gate_up_fused_bf16,
                                 seq_len, dim, 2 * intermediate);
        prof_ffn_gate_up_ms += dec_get_time_ms() - t0;

        t0 = dec_get_time_ms();
        qwen_swiglu_multiply(gate, gate_up, seq_len, intermediate);
        prof_ffn_swiglu_ms += dec_get_time_ms() - t0;

        t0 = dec_get_time_ms();
        qwen_linear_nobias_bf16(ffn_out, gate, l->down_weight_bf16,
                                 seq_len, intermediate, dim);
        prof_ffn_down_ms += dec_get_time_ms() - t0;

        t0 = dec_get_time_ms();
        qwen_add_inplace(x, ffn_out, seq_len * dim);
        prof_residual_ms += dec_get_time_ms() - t0;

    }

    ctx->kv_cache_len = start_pos + seq_len;

    if (qwen_verbose >= 2) {
        double total = prof_rms_norm_ms + prof_qkv_proj_ms + prof_qk_norm_rope_ms
                     + prof_kv_store_ms + prof_attn_ms + prof_out_proj_ms
                     + prof_residual_ms + prof_ffn_gate_up_ms + prof_ffn_swiglu_ms
                     + prof_ffn_down_ms;
        fprintf(stderr,
            "  [PREFILL PROFILE] seq=%d start_pos=%d layers=%d total=%.1fms\n"
            "    QKV proj:     %7.1fms (%4.1f%%)  [bf16->f32 + GEMM: Q(%dx%d->%d) K(%dx%d->%d) V(%dx%d->%d)]\n"
            "    Attn (causal): %6.1fms (%4.1f%%)  [%d q-tokens x %d kv-len, %d heads]\n"
            "    Out proj:     %7.1fms (%4.1f%%)  [bf16->f32 + GEMM: %dx%d->%d]\n"
            "    FFN gate_up:  %7.1fms (%4.1f%%)  [bf16->f32 + GEMM: %dx%d->%d]\n"
            "    FFN down:     %7.1fms (%4.1f%%)  [bf16->f32 + GEMM: %dx%d->%d]\n"
            "    RMSNorm:      %7.1fms (%4.1f%%)\n"
            "    QK norm+RoPE: %7.1fms (%4.1f%%)\n"
            "    KV store:     %7.1fms (%4.1f%%)\n"
            "    SwiGLU:       %7.1fms (%4.1f%%)\n"
            "    Residual add: %7.1fms (%4.1f%%)\n",
            seq_len, start_pos, cfg->dec_layers, total,
            prof_qkv_proj_ms, 100.0 * prof_qkv_proj_ms / total,
            seq_len, dim, q_dim, seq_len, dim, kv_dim, seq_len, dim, kv_dim,
            prof_attn_ms, 100.0 * prof_attn_ms / total,
            seq_len, start_pos + seq_len, n_heads,
            prof_out_proj_ms, 100.0 * prof_out_proj_ms / total,
            seq_len, q_dim, dim,
            prof_ffn_gate_up_ms, 100.0 * prof_ffn_gate_up_ms / total,
            seq_len, dim, 2 * intermediate,
            prof_ffn_down_ms, 100.0 * prof_ffn_down_ms / total,
            seq_len, intermediate, dim,
            prof_rms_norm_ms, 100.0 * prof_rms_norm_ms / total,
            prof_qk_norm_rope_ms, 100.0 * prof_qk_norm_rope_ms / total,
            prof_kv_store_ms, 100.0 * prof_kv_store_ms / total,
            prof_ffn_swiglu_ms, 100.0 * prof_ffn_swiglu_ms / total,
            prof_residual_ms, 100.0 * prof_residual_ms / total);
        qwen_linear_nobias_bf16_prof_print();
    }
}

/* ========================================================================
 * Decoder Forward (Single Token Generation)
 * ======================================================================== */

static void ensure_dec_buffers(qwen_ctx_t *ctx) {
    if (ctx->dec_x) return;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    int head_dim = cfg->dec_head_dim;

    ctx->dec_x        = (float *)malloc(dim * sizeof(float));
    ctx->dec_x_norm   = (float *)malloc(dim * sizeof(float));
    ctx->dec_q        = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_k        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(dim * sizeof(float));
    ctx->dec_gate     = (float *)malloc(2 * intermediate * sizeof(float));
    ctx->dec_up       = NULL; /* unused: gate buffer holds fused gate+up */
    ctx->dec_ffn_out  = (float *)malloc(dim * sizeof(float));
    ctx->dec_rope_cos = (float *)malloc(head_dim * sizeof(float));
    ctx->dec_rope_sin = (float *)malloc(head_dim * sizeof(float));
}

int qwen_decoder_forward(qwen_ctx_t *ctx, const float *input_embed) {
    qwen_decoder_t *dec = &ctx->decoder;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads;
    int n_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int intermediate = cfg->dec_intermediate;
    float eps = cfg->dec_rms_norm_eps;
    float theta = cfg->dec_rope_theta;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    ensure_dec_buffers(ctx);
    float *x = ctx->dec_x;
    float *x_norm = ctx->dec_x_norm;
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;
    float *attn_out = ctx->dec_attn_out;
    float *proj_out = ctx->dec_proj_out;
    float *gate_buf = ctx->dec_gate;
    float *ffn_out = ctx->dec_ffn_out;
    memcpy(x, input_embed, dim * sizeof(float));

    int pos = ctx->kv_cache_len;

    /* Grow KV cache if needed */
    if (pos >= ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, pos + 1024) != 0) return QWEN_TOKEN_IM_END;
    }

    if (ensure_rope_cache(ctx, pos + 1, head_dim, theta) != 0) {
        return QWEN_TOKEN_IM_END;
    }
    const float *rope_cos = ctx->rope_cache_cos + (size_t)pos * head_dim;
    const float *rope_sin = ctx->rope_cache_sin + (size_t)pos * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    int n_layers_fwd = cfg->dec_layers;
    if (ctx->dec_layers_limit > 0 && ctx->dec_layers_limit < n_layers_fwd)
        n_layers_fwd = ctx->dec_layers_limit;

    for (int layer = 0; layer < n_layers_fwd; layer++) {
        qwen_dec_layer_t *l = &dec->layers[layer];

        qwen_rms_norm(x_norm, x, l->input_norm, 1, dim, eps);
        qwen_linear_nobias_bf16_qkv(q, k, v, x_norm,
                                    l->wq_weight_bf16,
                                    l->wk_weight_bf16,
                                    l->wv_weight_bf16,
                                    dim, q_dim, kv_dim);

        /* Per-head Q/K RMSNorm */
        qwen_rms_norm_per_head(q, l->q_norm_weight, 1, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k, l->k_norm_weight, 1, n_kv_heads, head_dim, eps);

        /* Apply NeoX RoPE */
        qwen_apply_rope_neox(q, rope_cos, rope_sin, 1, n_heads, head_dim);
        qwen_apply_rope_neox(k, rope_cos, rope_sin, 1, n_kv_heads, head_dim);

        memcpy(kv_cache_k_at(ctx, layer, pos), k, kv_dim * sizeof(float));
        memcpy(kv_cache_v_at(ctx, layer, pos), v, kv_dim * sizeof(float));

        int total_seq = pos + 1;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        qwen_causal_attention(attn_out, q, full_k, full_v,
                               1, total_seq, n_heads, n_kv_heads,
                               head_dim, scale, pos);

        qwen_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, 1, q_dim, dim);
        qwen_add_inplace(x, proj_out, dim);

        qwen_rms_norm(x_norm, x, l->post_attn_norm, 1, dim, eps);

        /* Fused gate+up matvec: one pass over x_norm, output interleaved [g0,u0,g1,u1,...] */
        qwen_linear_nobias_bf16(gate_buf, x_norm, l->gate_up_fused_bf16,
                                 1, dim, 2 * intermediate);
        /* In-place for seq=1: gate_buf[0:inter] receives SwiGLU output. */
        qwen_swiglu_multiply(gate_buf, gate_buf, 1, intermediate);
        qwen_linear_nobias_bf16(ffn_out, gate_buf, l->down_weight_bf16, 1, intermediate, dim);
        qwen_add_inplace(x, ffn_out, dim);
    }

    ctx->kv_cache_len = pos + 1;

    /* Final norm + streaming argmax (no logits buffer needed) */
    qwen_rms_norm(x, x, dec->norm, 1, dim, eps);
    return qwen_argmax_matvec_bf16(x, dec->tok_embeddings_bf16, dim, cfg->vocab_size);
}
