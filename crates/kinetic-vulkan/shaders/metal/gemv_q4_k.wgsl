// block_q4_K: d, dmin, scales[12], qs[128]; 144 bytes, nine words.

// Invocation j's shares of block `blk` of up to four rows, `n` blocks
// apart, for input block b. The input is loaded and made float once for all
// the rows; each row's share is then computed as `k4_combine` computes it.
//
// Each sum is of 16 products of a code below 16 and an int8, exact in f32,
// so it is the integer sum bit for bit; a GPU without accelerated int8 dot
// products multiplies floats faster.
fn tile(total: u32, blk: u32, n: u32, live: u32, b: u32, j: u32) -> vec4<f32> {
    let s = 2u * (j / 2u);
    let xi = k4_input(b, j);
    let u = xq[xi];
    let v = xq[xi + 2u];
    let x0 = xd[b * 8u + s];
    let x1 = xd[b * 8u + s + 1u];
    let u0 = vec4<f32>(unpack4xI8(u.x));
    let u1 = vec4<f32>(unpack4xI8(u.y));
    let u2 = vec4<f32>(unpack4xI8(u.z));
    let u3 = vec4<f32>(unpack4xI8(u.w));
    let v0 = vec4<f32>(unpack4xI8(v.x));
    let v1 = vec4<f32>(unpack4xI8(v.y));
    let v2 = vec4<f32>(unpack4xI8(v.z));
    let v3 = vec4<f32>(unpack4xI8(v.w));
    var out = vec4<f32>(0.0);
    for (var r = 0u; r < live; r++) {
        let at = (blk + r * n) * 9u;
        let head = w[at];
        let q = w[at + 1u + j];
        let l = q & vec4<u32>(NIBBLES);
        let h = (q >> vec4<u32>(4u)) & vec4<u32>(NIBBLES);
        let lo = dot(vec4<f32>(unpack4xU8(l.x)), u0) + dot(vec4<f32>(unpack4xU8(l.y)), u1)
            + dot(vec4<f32>(unpack4xU8(l.z)), u2) + dot(vec4<f32>(unpack4xU8(l.w)), u3);
        let hi = dot(vec4<f32>(unpack4xU8(h.x)), v0) + dot(vec4<f32>(unpack4xU8(h.y)), v1)
            + dot(vec4<f32>(unpack4xU8(h.z)), v2) + dot(vec4<f32>(unpack4xU8(h.w)), v3);
        let a = k4_scale_min(s, head.y, head.z, head.w);
        let c = k4_scale_min(s + 1u, head.y, head.z, head.w);
        let dm = unpack2x16float(head.x);
        var share = dm.x * (a.x * x0.x * lo + c.x * x1.x * hi);
        if j % 2u == 0u {
            share -= dm.y * (a.y * x0.y + c.y * x1.y);
        }
        out[r] = share;
    }
    return out;
}
