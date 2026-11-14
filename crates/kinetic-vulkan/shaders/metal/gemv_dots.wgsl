// `tile` for formats that define one row's share as `block_dot`, which
// loads its input itself.
fn tile(total: u32, blk: u32, n: u32, live: u32, b: u32, part: u32) -> vec4<f32> {
    var out = vec4<f32>(0.0);
    for (var r = 0u; r < live; r++) {
        out[r] = block_dot(total, blk + r * n, b, part);
    }
    return out;
}
