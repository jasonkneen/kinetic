// threadgroup bytes: 
// y = W x for the first `batch[0]` rows x of the input, W in Q5_K: what
// the GEMV kernel of `gemv.wgsl`, `k4.wgsl` and `gemv_q5_k.wgsl` computes,
// bit for bit, written for a pass of many tokens. The Q5_K form of
// `gemv_q4_k_many.metal`; the notes below are that kernel's. Hand-written: what makes it fast
// is what stays in registers, which WGSL leaves to its translation.
//
// The GEMV kernel unpacks a row's weights again for every token. Here a
// SIMD group takes two rows; per weight block it unpacks them once, as
// halves, which hold a code below 16 exactly in half the registers, and
// then runs TOKENS tokens against them, loading each token's block once
// for both rows. More rows or more tokens no longer fit the registers and
// run slower. For a token or two the GEMV kernel is the faster one.
//
// Every output is the sum it is there. An invocation's share of a block is
// `k4_combine` of two sums of 16 products of a code below 32 and an int8,
// which are integers below 2^24 and so the same floats in any order; the shares are
// added per invocation in block order, and the 32 invocations by
// `simd_sum`, as there.
#include <metal_stdlib>
using namespace metal;

// Rows of one SIMD group and tokens of one run over the weights.
#define TILE 2
#define TOKENS 16
#define NIBBLES 0x0F0F0F0Fu

struct Shape {
    uint rows;
    uint blocks_per_row;
    uint stride;
    uint offset;
    uint head;
};

// As in `k4.wgsl`.
static inline float2 k4_scale_min(uint s, uint q0, uint q1, uint q2)
{
    if (s < 4u) {
        return float2(float((q0 >> (8u * s)) & 63u), float((q1 >> (8u * s)) & 63u));
    }
    const uint t = 8u * (s - 4u);
    return float2(float(((q2 >> t) & 15u) | (((q0 >> t) >> 2u) & 48u)),
                  float((((q2 >> t) >> 4u) & 15u) | (((q1 >> t) >> 2u) & 48u)));
}

[[max_total_threads_per_threadgroup(128)]] kernel void main_(
    device const uint4 *w [[buffer(0)]],
    device const uint4 *xq [[buffer(2)]],
    device const float4 *xd [[buffer(3)]],
    device float *y [[buffer(4)]],
    device const uint *batch [[buffer(5)]],
    constant Shape &shape [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]],
    uint slot [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = ((group.y * groups.x) + group.x) * (4u * TILE) + slot * TILE;
    if (row >= shape.rows) {
        return;
    }
    // The same for the whole SIMD group, as everything that depends on it.
    const uint live = min(uint(TILE), shape.rows - row);
    const uint count = batch[0];
    const uint n = shape.blocks_per_row;
    const uint out = shape.offset + row;
    // Invocation j of a block takes 16 elements of sub-block `sub` from the
    // low nibbles of its word and the same 16 of the next from the high.
    const uint j = lane % 8u;
    const uint sub = 2u * (j / 2u);
    const bool takes_min = (j % 2u) == 0u;
    for (uint t0 = 0; t0 < count; t0 += TOKENS) {
        float sum[TOKENS][TILE];
        for (uint t = 0; t < TOKENS; t++) {
            for (uint r = 0; r < TILE; r++) {
                sum[t][r] = 0.0f;
            }
        }
        for (uint b = lane / 8u; b < n; b += 4u) {
            half4 lows[TILE][4], highs[TILE][4];
            float d[TILE], dmin[TILE], scale0[TILE], min0[TILE], scale1[TILE], min1[TILE];
            for (uint r = 0; r < live; r++) {
                // block_q5_K: d, dmin, scales[12], qh[32], qs[128]; eleven
                // words. Bit t of qh[l] is the fifth bit of element l of
                // sub-block t.
                const uint at = ((row + r) * n + b) * 11u;
                const uint4 head = w[at];
                const uint4 qh = w[(at + 1u) + (j % 2u)];
                const uint4 q = w[(at + 3u) + j];
                const uint4 ones = uint4(0x01010101u);
                const uint4 l = (q & uint4(NIBBLES)) | (((qh >> uint4(sub)) & ones) << uint4(4u));
                const uint4 h = ((q >> uint4(4u)) & uint4(NIBBLES))
                                | (((qh >> uint4(sub + 1u)) & ones) << uint4(4u));
                lows[r][0] = half4(uint4(as_type<packed_uchar4>(l.x)));
                lows[r][1] = half4(uint4(as_type<packed_uchar4>(l.y)));
                lows[r][2] = half4(uint4(as_type<packed_uchar4>(l.z)));
                lows[r][3] = half4(uint4(as_type<packed_uchar4>(l.w)));
                highs[r][0] = half4(uint4(as_type<packed_uchar4>(h.x)));
                highs[r][1] = half4(uint4(as_type<packed_uchar4>(h.y)));
                highs[r][2] = half4(uint4(as_type<packed_uchar4>(h.z)));
                highs[r][3] = half4(uint4(as_type<packed_uchar4>(h.w)));
                const float2 a = k4_scale_min(sub, head.y, head.z, head.w);
                const float2 c = k4_scale_min(sub + 1u, head.y, head.z, head.w);
                const float2 dm = float2(as_type<half2>(head.x));
                d[r] = dm.x;
                dmin[r] = dm.y;
                scale0[r] = a.x;
                min0[r] = a.y;
                scale1[r] = c.x;
                min1[r] = c.y;
            }
            for (uint t = 0; t < TOKENS && t0 + t < count; t++) {
                const uint xb = (t0 + t) * n + b;
                const uint xi = ((xb * 16u) + (4u * (j / 2u))) + (j % 2u);
                const uint4 u = xq[xi];
                const uint4 v = xq[xi + 2u];
                // (d, d * sum) of both sub-blocks of the input in one load.
                const float4 x01 = xd[((xb * 8u) + sub) / 2u];
                const float2 x0 = x01.xy;
                const float2 x1 = x01.zw;
                const float4 u0 = float4(int4(as_type<packed_char4>(u.x)));
                const float4 u1 = float4(int4(as_type<packed_char4>(u.y)));
                const float4 u2 = float4(int4(as_type<packed_char4>(u.z)));
                const float4 u3 = float4(int4(as_type<packed_char4>(u.w)));
                const float4 v0 = float4(int4(as_type<packed_char4>(v.x)));
                const float4 v1 = float4(int4(as_type<packed_char4>(v.y)));
                const float4 v2 = float4(int4(as_type<packed_char4>(v.z)));
                const float4 v3 = float4(int4(as_type<packed_char4>(v.w)));
                for (uint r = 0; r < live; r++) {
                    const float lo = ((dot(float4(lows[r][0]), u0) + dot(float4(lows[r][1]), u1))
                                      + dot(float4(lows[r][2]), u2))
                                     + dot(float4(lows[r][3]), u3);
                    const float hi = ((dot(float4(highs[r][0]), v0) + dot(float4(highs[r][1]), v1))
                                      + dot(float4(highs[r][2]), v2))
                                     + dot(float4(highs[r][3]), v3);
                    // `k4_combine`, term for term.
                    float share = d[r] * (((scale0[r] * x0.x) * lo) + ((scale1[r] * x1.x) * hi));
                    if (takes_min) {
                        share = share - (dmin[r] * ((min0[r] * x0.y) + (min1[r] * x1.y)));
                    }
                    sum[t][r] = sum[t][r] + share;
                }
            }
        }
        for (uint t = 0; t < TOKENS && t0 + t < count; t++) {
            for (uint r = 0; r < live; r++) {
                const float all = simd_sum(sum[t][r]);
                if (lane == 0u) {
                    y[(((t0 + t) * shape.stride) + out) + r] = all;
                }
            }
        }
    }
}
