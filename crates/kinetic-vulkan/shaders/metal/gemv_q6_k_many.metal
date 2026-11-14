// threadgroup bytes: 
// y = W x for the first `batch[0]` rows x of the input, W in Q6_K: what
// the GEMV kernel of `gemv.wgsl` and `gemv_q6_k.wgsl` computes, bit for
// bit, written for a pass of many tokens. See `gemv_q4_k_many.metal`, of
// which this is the Q6_K form: two rows per SIMD group, unpacked once per
// weight block as halves, which hold a quant of -32..31 exactly, and
// TOKENS tokens run against them.
//
// Every output is the sum it is there. An invocation's share of a block is
// built from two sums of 16 products of a quant and an int8, which are
// integers below 2^24 and so the same floats in any order; the rest is
// term for term what `tile` does there.
#include <metal_stdlib>
using namespace metal;

// Metal's compiler fuses products into sums where it likes. Here it may
// not, and the one fusion it makes in the GEMV kernel is written out below:
// which that is cannot be read off the source, so
// `q6_k_prompt_pass_has_the_same_bits` compares the two kernels bit for bit.
#pragma STDC FP_CONTRACT OFF

// Rows of one SIMD group and tokens of one run over the weights.
#define TILE 2
#define TOKENS 16
#define NIBBLES 0x0F0F0F0Fu
#define LOW2 0x03030303u

struct Shape {
    uint rows;
    uint blocks_per_row;
    uint stride;
    uint offset;
    uint head;
};

// Signed byte `i` of a word of 16.
static inline float sbyte(uint4 v, uint i)
{
    return float((int(v[i / 4u]) << (24u - 8u * (i % 4u))) >> 24u);
}

[[max_total_threads_per_threadgroup(128)]] kernel void main_(
    device const uint4 *w [[buffer(0)]],
    device const uint4 *xq [[buffer(2)]],
    device const float2 *xd [[buffer(3)]],
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
    const uint blocks = shape.blocks_per_row;
    const uint total = shape.rows * blocks;
    const uint out = shape.offset + row;
    // Invocation j = 4 half + k of a block, as in `gemv_q6_k.wgsl`.
    const uint j = lane % 8u;
    const uint half_of_block = j / 4u;
    const uint k = j % 4u;
    const uint g = k / 2u;
    const uint odd = k & 1u;
    const uint sc = ((8u * half_of_block) + odd) + (2u * g);
    for (uint t0 = 0; t0 < count; t0 += TOKENS) {
        float sum[TOKENS][TILE];
        for (uint t = 0; t < TOKENS; t++) {
            for (uint r = 0; r < TILE; r++) {
                sum[t][r] = 0.0f;
            }
        }
        for (uint b = lane / 8u; b < blocks; b += 4u) {
            half4 lows[TILE][4], highs[TILE][4];
            float scale0[TILE], scale1[TILE], d[TILE];
            for (uint r = 0; r < live; r++) {
                const uint at = (row + r) * blocks + b;
                const uint4 lq = w[(at * 8u) + j];
                const uint4 hq = w[(((total * 8u) + (at * 4u)) + (2u * half_of_block)) + odd];
                const uint4 l = (lq & uint4(NIBBLES))
                                | (((hq >> uint4(2u * g)) & uint4(LOW2)) << uint4(4u));
                const uint4 h = ((lq >> uint4(4u)) & uint4(NIBBLES))
                                | (((hq >> uint4((2u * g) + 4u)) & uint4(LOW2)) << uint4(4u));
                // Quants are stored as q + 32.
                lows[r][0] = half4(int4(uint4(as_type<packed_uchar4>(l.x))) - 32);
                lows[r][1] = half4(int4(uint4(as_type<packed_uchar4>(l.y))) - 32);
                lows[r][2] = half4(int4(uint4(as_type<packed_uchar4>(l.z))) - 32);
                lows[r][3] = half4(int4(uint4(as_type<packed_uchar4>(l.w))) - 32);
                highs[r][0] = half4(int4(uint4(as_type<packed_uchar4>(h.x))) - 32);
                highs[r][1] = half4(int4(uint4(as_type<packed_uchar4>(h.y))) - 32);
                highs[r][2] = half4(int4(uint4(as_type<packed_uchar4>(h.z))) - 32);
                highs[r][3] = half4(int4(uint4(as_type<packed_uchar4>(h.w))) - 32);
                const uint4 scales = w[(total * 12u) + at];
                scale0[r] = sbyte(scales, sc);
                scale1[r] = sbyte(scales, sc + 4u);
                const float2 pair = float2(as_type<half2>(w[(total * 13u) + (at / 8u)][(at / 2u) % 4u]));
                d[r] = ((at & 1u) == 1u) ? pair.y : pair.x;
            }
            for (uint t = 0; t < TOKENS && t0 + t < count; t++) {
                const uint e = (((((t0 + t) * blocks + b) * 256u) + (128u * half_of_block))
                                + (32u * g))
                               + (16u * odd);
                const uint4 u = xq[e / 16u];
                const uint4 v = xq[(e / 16u) + 4u];
                const float d0 = xd[e / 32u].x;
                const float d1 = xd[(e / 32u) + 2u].x;
                const float4 u0 = float4(int4(as_type<packed_char4>(u.x)));
                const float4 u1 = float4(int4(as_type<packed_char4>(u.y)));
                const float4 u2 = float4(int4(as_type<packed_char4>(u.z)));
                const float4 u3 = float4(int4(as_type<packed_char4>(u.w)));
                const float4 v0 = float4(int4(as_type<packed_char4>(v.x)));
                const float4 v1 = float4(int4(as_type<packed_char4>(v.y)));
                const float4 v2 = float4(int4(as_type<packed_char4>(v.z)));
                const float4 v3 = float4(int4(as_type<packed_char4>(v.w)));
                for (uint r = 0; r < live; r++) {
                    const float s0 = ((dot(float4(lows[r][0]), u0) + dot(float4(lows[r][1]), u1))
                                      + dot(float4(lows[r][2]), u2))
                                     + dot(float4(lows[r][3]), u3);
                    const float s1 = ((dot(float4(highs[r][0]), v0) + dot(float4(highs[r][1]), v1))
                                      + dot(float4(highs[r][2]), v2))
                                     + dot(float4(highs[r][3]), v3);
                    // `tile`, as Metal compiles it there.
                    const float share = fma(scale0[r] * d0, s0, (scale1[r] * d1) * s1);
                    sum[t][r] = sum[t][r] + (d[r] * share);
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
