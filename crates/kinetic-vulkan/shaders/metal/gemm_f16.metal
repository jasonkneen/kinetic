// threadgroup bytes: 8192 4096
// y[t][offset + r] = sum over k of w[r][k] x[t][k], for the tokens of a
// prompt batch and every row r of w, with Metal's SIMD-group matrices,
// which the GPU multiplies 8x8 at a time. Hand-written: WGSL has no such
// type.
//
// A threadgroup takes 64 rows and 32 tokens. Per 32 columns its threads
// first copy that part of w and of x to threadgroup memory, each thread a
// run of consecutive values, laid out as the 8x8 tiles the product takes.
// Each of the four SIMD groups then multiplies four tiles of rows by two
// of tokens. Loading the tiles from the device's memory instead reads
// eight rows at a time, a few values of each, and is slower.
//
// One pass runs as many tokens as the batch holds, and the more the
// better: the weights are read once however many there are. Token groups
// past the batch's count return at once; the last one may run tokens past
// it, on whatever their rows hold, into rows nobody reads. A last
// threadgroup with fewer than 64 rows left takes zeros for the rest and
// writes its sums one by one, the rows that exist only.
//
// The weights are f16 and the sums f32. Unlike the GEMV kernels this does
// not give a token the sum it gets alone, so it only runs prompt batches
// (`Commands::set_prompt`). A sum is the same whenever it is computed:
// the order of its terms depends on the matrix alone.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct Shape {
    uint rows;
    uint cols;
    uint stride;
    uint offset;
};

[[max_total_threads_per_threadgroup(128)]] kernel void main_(
    device const half *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    device const uint *batch [[buffer(3)]],
    constant Shape &s [[buffer(4)]],
    threadgroup float *rows_part [[threadgroup(0)]],
    threadgroup float *tokens_part [[threadgroup(1)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const ulong r0 = (ulong)group.x * 64;
    const ulong t0 = (ulong)group.y * 32;
    if (t0 >= batch[0]) {
        return;
    }
    simdgroup_float8x8 sum[8];
    for (int i = 0; i < 8; i++) {
        sum[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // A thread copies 16 weights of one row, two per row, and 8 values of
    // one token, four per token.
    const ushort row = tid / 2, half_of_row = tid % 2;
    const ushort token = tid / 4, quarter = tid % 4;
    const bool has_row = r0 + row < s.rows;
    device const half *from_w = w + (r0 + row) * s.cols + 16 * half_of_row;
    device const float *from_x = x + (t0 + token) * s.cols + 8 * quarter;
    // Tiles by step of eight columns, then by eight rows or tokens; a tile
    // holds a row or a token per eight floats.
    threadgroup float *to_rows =
        rows_part + (2 * half_of_row * 8 + row / 8) * 64 + (row % 8) * 8;
    threadgroup float *to_tokens = tokens_part + (quarter * 4 + token / 8) * 64 + (token % 8) * 8;
    // This SIMD group's half of the rows and half of the tokens.
    threadgroup const float *my_rows = rows_part + 4 * 64 * (simd_group % 2);
    threadgroup const float *my_tokens = tokens_part + 2 * 64 * (simd_group / 2);
    simdgroup_float8x8 a[4];
    simdgroup_float8x8 b[2];
    for (uint k = 0; k < s.cols; k += 32) {
        // Nobody still reads the last part.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 w0 = float4(0.0f), w1 = float4(0.0f), w2 = float4(0.0f), w3 = float4(0.0f);
        if (has_row) {
            w0 = float4(*(device const packed_half4 *)(from_w + k));
            w1 = float4(*(device const packed_half4 *)(from_w + k + 4));
            w2 = float4(*(device const packed_half4 *)(from_w + k + 8));
            w3 = float4(*(device const packed_half4 *)(from_w + k + 12));
        }
        *(threadgroup packed_float4 *)(to_rows) = w0;
        *(threadgroup packed_float4 *)(to_rows + 4) = w1;
        *(threadgroup packed_float4 *)(to_rows + 512) = w2;
        *(threadgroup packed_float4 *)(to_rows + 516) = w3;
        *(threadgroup packed_float4 *)(to_tokens) = *(device const packed_float4 *)(from_x + k);
        *(threadgroup packed_float4 *)(to_tokens + 4) =
            *(device const packed_float4 *)(from_x + k + 4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int step = 0; step < 4; step++) {
            for (int i = 0; i < 4; i++) {
                simdgroup_load(a[i], my_rows + step * 512 + 64 * i, 8, 0, false);
            }
            // A tile holds a token per row; the product wants one per
            // column.
            for (int i = 0; i < 2; i++) {
                simdgroup_load(b[i], my_tokens + step * 256 + 64 * i, 8, 0, true);
            }
            for (int i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(sum[i], a[i % 4], b[i / 4], sum[i]);
            }
        }
    }
    const ulong first_row = r0 + 32 * (simd_group % 2);
    const ulong first_token = t0 + 16 * (simd_group / 2);
    if (r0 + 64 <= s.rows) {
        for (int i = 0; i < 8; i++) {
            simdgroup_store(sum[i], y + s.offset, s.stride,
                            ulong2(first_row + 8 * (i % 4), first_token + 8 * (i / 4)), true);
        }
        return;
    }
    // Nobody still reads the rows' part, which holds just the eight tiles
    // of each SIMD group: a row of the matrix per eight floats.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *mine = rows_part + 512 * simd_group;
    for (int i = 0; i < 8; i++) {
        simdgroup_store(sum[i], mine + 64 * i, 8, 0, false);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 512; e += 32) {
        const ulong at_row = first_row + 8 * ((e / 64) % 4) + (e % 64) / 8;
        const ulong at_token = first_token + 8 * (e / 256) + e % 8;
        if (at_row < s.rows) {
            y[at_token * s.stride + s.offset + at_row] = mine[e];
        }
    }
}
