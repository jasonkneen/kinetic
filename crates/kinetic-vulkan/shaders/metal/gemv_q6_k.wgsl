// block_q6_K is 210 bytes, so blocks are only 2-byte aligned. The device copy
// is planar: all ql[128], then all qh[64], all scales[16], all f16 d.
//
// A block is two halves of 128 elements; in half n, element 32g + l (g < 4,
// l < 32) takes its low four bits from ql[64n + l + 32(g & 1)] (low nibble
// for g < 2, high for g >= 2) and its top two from bits 2g..2g+1 of
// qh[32n + l]. Each run of 16 elements has its own int8 scale.
//
// Eight invocations share a block. Invocation j = 4n + k loads ql bytes
// 64n + 16k, which hold elements l = 16(k & 1) .. +16 of group g = k / 2
// (low nibbles) and of group g + 2 (high nibbles).

const LANES: u32 = 8u;

// Signed byte `i` of a word.
fn sbyte(v: vec4<u32>, i: u32) -> f32 {
    return f32(extractBits(bitcast<i32>(v[i / 4u]), 8u * (i % 4u), 8u));
}

// Invocation j's shares of block `blk` of up to four rows, `n` blocks
// apart, for input block b. The input is loaded and made float once for all
// the rows. Quants are stored as q + 32, taken off as they become float.
//
// Each sum is of 16 products of a code below 32 in size and an int8, exact
// in f32, so it is the integer sum bit for bit; a GPU without accelerated
// int8 dot products multiplies floats faster.
fn tile(total: u32, blk: u32, rows_apart: u32, live: u32, b: u32, j: u32) -> vec4<f32> {
    let n = j / 4u;
    let k = j % 4u;
    let g = k / 2u;
    let e = b * 256u + 128u * n + 32u * g + 16u * (k & 1u);
    let u = xq[e / 16u];
    let v = xq[e / 16u + 4u];
    let d0 = xd[e / 32u].x;
    let d1 = xd[e / 32u + 2u].x;
    let u0 = vec4<f32>(unpack4xI8(u.x));
    let u1 = vec4<f32>(unpack4xI8(u.y));
    let u2 = vec4<f32>(unpack4xI8(u.z));
    let u3 = vec4<f32>(unpack4xI8(u.w));
    let v0 = vec4<f32>(unpack4xI8(v.x));
    let v1 = vec4<f32>(unpack4xI8(v.y));
    let v2 = vec4<f32>(unpack4xI8(v.z));
    let v3 = vec4<f32>(unpack4xI8(v.w));
    let low2 = vec4<u32>(0x03030303u);
    let off = vec4<f32>(32.0);
    let sc = 8u * n + (k & 1u) + 2u * g;
    var out = vec4<f32>(0.0);
    for (var r = 0u; r < live; r++) {
        let at = blk + r * rows_apart;
        let lq = w[at * 8u + j];
        let hq = w[total * 8u + at * 4u + 2u * n + (k & 1u)];
        let l = (lq & vec4<u32>(NIBBLES)) | (((hq >> vec4<u32>(2u * g)) & low2) << vec4<u32>(4u));
        let h = ((lq >> vec4<u32>(4u)) & vec4<u32>(NIBBLES))
            | (((hq >> vec4<u32>(2u * g + 4u)) & low2) << vec4<u32>(4u));
        let s0 = dot(vec4<f32>(unpack4xU8(l.x)) - off, u0) + dot(vec4<f32>(unpack4xU8(l.y)) - off, u1)
            + dot(vec4<f32>(unpack4xU8(l.z)) - off, u2) + dot(vec4<f32>(unpack4xU8(l.w)) - off, u3);
        let s1 = dot(vec4<f32>(unpack4xU8(h.x)) - off, v0) + dot(vec4<f32>(unpack4xU8(h.y)) - off, v1)
            + dot(vec4<f32>(unpack4xU8(h.z)) - off, v2) + dot(vec4<f32>(unpack4xU8(h.w)) - off, v3);
        let scales = w[total * 12u + at];
        // The fusion Metal's compiler chose for this sum, written out so
        // that `gemv_q6_k_many.metal` can match it on any compiler.
        let share = fma(sbyte(scales, sc) * d0, s0, sbyte(scales, sc + 4u) * d1 * s1);
        out[r] = f16_at(total * 13u, at) * share;
    }
    return out;
}
