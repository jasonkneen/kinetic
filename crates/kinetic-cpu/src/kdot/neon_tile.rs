//! NEON tile kernels: a few unpacked weight rows times eight interleaved
//! tokens, or four or two for short batches.
//!
//! SDOT multiplies four bytes of each lane by four bytes of the other
//! operand and adds the four products into that lane. Each step broadcasts
//! four weight bytes of one row against four values of four tokens, so a
//! lane of the result is one token; two such products cover eight tokens.
//! Narrower groups broadcast eight or sixteen bytes against as many values
//! of four or two tokens, and each token's lanes are added once per block.
//! The sums of one 16-value scale group are exact i32 before the scale
//! multiplies them, and the f32 steps are exactly those of the one-row
//! kernels, so a tile gives the same value for every row and token.

use std::arch::aarch64::*;

use super::neon::sdot;
use super::{half, scales_mins, TileKernel, Unpacked, Unpacked32, TILE_ROWS};
use crate::q8::Q8Group;
use crate::q8k::{Q8KGroup, GROUP};
use kinetic_quant::TensorType;

// An [`Unpacked`] block here holds the unsigned codes in element order in
// `q`, the i32 scale of each 16-value group in `sc`, and in `pairs` the
// factors applied to the activation sums as [`pair`]s: the mins of each
// pair of 32-value sub-blocks for Q4_K and Q5_K, the scales of each pair
// of 16-value groups for Q6_K, which stores its weights as `q + 32`.

pub fn kernel(ty: TensorType) -> Option<TileKernel<Unpacked, Q8KGroup>> {
    // SAFETY (each closure): only reachable after `neon::available()`
    // confirmed the target features the implementations are compiled for.
    match ty {
        TensorType::Q4K => Some(TileKernel {
            unpack: |row, out| unsafe { unpack_q4_k(row, out) },
            dot4: |rows, x| unsafe { tiles::<TILE_ROWS, true>(rows, x) },
            dot1: |row, x| unsafe { tiles::<1, true>([row], x)[0] },
        }),
        TensorType::Q5K => Some(TileKernel {
            unpack: |row, out| unsafe { unpack_q5_k(row, out) },
            dot4: |rows, x| unsafe { tiles::<TILE_ROWS, true>(rows, x) },
            dot1: |row, x| unsafe { tiles::<1, true>([row], x)[0] },
        }),
        TensorType::Q6K => Some(TileKernel {
            unpack: |row, out| unsafe { unpack_q6_k(row, out) },
            dot4: |rows, x| unsafe { tiles::<TILE_ROWS, false>(rows, x) },
            dot1: |row, x| unsafe { tiles::<1, false>([row], x)[0] },
        }),
        _ => None,
    }
}

pub fn kernel32(ty: TensorType) -> Option<TileKernel<Unpacked32, Q8Group>> {
    // SAFETY (each closure): as in `kernel`.
    match ty {
        TensorType::Q8_0 => Some(TileKernel {
            unpack: |row, out| unsafe { unpack_q8_0(row, out) },
            dot4: |rows, x| unsafe { tiles32::<TILE_ROWS, false>(rows, x) },
            dot1: |row, x| unsafe { tiles32::<1, false>([row], x)[0] },
        }),
        TensorType::Q4_0 => Some(TileKernel {
            unpack: |row, out| unsafe { unpack_q4_0(row, out) },
            dot4: |rows, x| unsafe { tiles32::<TILE_ROWS, true>(rows, x) },
            dot1: |row, x| unsafe { tiles32::<1, true>([row], x)[0] },
        }),
        _ => None,
    }
}

/// `lo` and `hi` as the low and high i16 halves of one i32.
fn pair(lo: i16, hi: i16) -> i32 {
    (lo as u16 as i32) | ((hi as i32) << 16)
}

#[inline(always)]
unsafe fn store(p: *mut u8, v: uint8x16_t) {
    vst1q_u8(p, v);
}

/// The scales of eight 32-value sub-blocks into the 16 group slots, and
/// their mins into the pairs.
fn set_k_scales(u: &mut Unpacked, sc: &[u8; 8], m: &[u8; 8]) {
    for (g, &s) in sc.iter().enumerate() {
        u.sc[2 * g] = s as i32;
        u.sc[2 * g + 1] = s as i32;
    }
    for p in 0..4 {
        u.pairs[p] = pair(m[2 * p] as i16, m[2 * p + 1] as i16);
    }
    u.pairs[4..].fill(0);
}

#[target_feature(enable = "neon")]
unsafe fn unpack_q4_k(row: &[u8], out: &mut [Unpacked]) {
    let low = vdupq_n_u8(0x0F);
    for (blk, u) in row.chunks_exact(144).zip(out) {
        (u.d, u.dmin) = (half(&blk[0..2]), half(&blk[2..4]));
        let (sc, m) = scales_mins(&blk[4..16]);
        let qs = blk[16..].as_ptr();
        let q = u.q.as_mut_ptr();
        for j in 0..4 {
            let q0 = vld1q_u8(qs.add(32 * j));
            let q1 = vld1q_u8(qs.add(32 * j + 16));
            store(q.add(64 * j), vandq_u8(q0, low));
            store(q.add(64 * j + 16), vandq_u8(q1, low));
            store(q.add(64 * j + 32), vshrq_n_u8::<4>(q0));
            store(q.add(64 * j + 48), vshrq_n_u8::<4>(q1));
        }
        set_k_scales(u, &sc, &m);
    }
}

#[target_feature(enable = "neon")]
unsafe fn unpack_q5_k(row: &[u8], out: &mut [Unpacked]) {
    let low = vdupq_n_u8(0x0F);
    let bit4 = vdupq_n_u8(0x10);
    for (blk, u) in row.chunks_exact(176).zip(out) {
        (u.d, u.dmin) = (half(&blk[0..2]), half(&blk[2..4]));
        let (sc, m) = scales_mins(&blk[4..16]);
        // Bit t of each qh byte is the fifth bit of sub-block t, as in the
        // one-row kernel.
        let qh = [vld1q_u8(blk[16..].as_ptr()), vld1q_u8(blk[32..].as_ptr())];
        let fifth =
            |qh: uint8x16_t, t: usize| vandq_u8(vshlq_u8(qh, vdupq_n_s8(4 - t as i8)), bit4);
        let qs = blk[48..].as_ptr();
        let q = u.q.as_mut_ptr();
        for j in 0..4 {
            let q0 = vld1q_u8(qs.add(32 * j));
            let q1 = vld1q_u8(qs.add(32 * j + 16));
            store(
                q.add(64 * j),
                vorrq_u8(vandq_u8(q0, low), fifth(qh[0], 2 * j)),
            );
            store(
                q.add(64 * j + 16),
                vorrq_u8(vandq_u8(q1, low), fifth(qh[1], 2 * j)),
            );
            store(
                q.add(64 * j + 32),
                vorrq_u8(vshrq_n_u8::<4>(q0), fifth(qh[0], 2 * j + 1)),
            );
            store(
                q.add(64 * j + 48),
                vorrq_u8(vshrq_n_u8::<4>(q1), fifth(qh[1], 2 * j + 1)),
            );
        }
        set_k_scales(u, &sc, &m);
    }
}

/// Bits `shift..shift + 2` of each qh byte, moved to bits 4..6.
#[inline(always)]
unsafe fn q6_high(qh: uint8x16_t, shift: i8) -> uint8x16_t {
    vandq_u8(vshlq_u8(qh, vdupq_n_s8(4 - shift)), vdupq_n_u8(0x30))
}

#[target_feature(enable = "neon")]
unsafe fn unpack_q6_k(row: &[u8], out: &mut [Unpacked]) {
    let low = vdupq_n_u8(0x0F);
    for (blk, u) in row.chunks_exact(210).zip(out) {
        let q = u.q.as_mut_ptr();
        for n in 0..2 {
            let ql = blk[64 * n..].as_ptr();
            let qh = blk[128 + 32 * n..].as_ptr();
            for h in 0..2 {
                let ql0 = vld1q_u8(ql.add(16 * h));
                let ql1 = vld1q_u8(ql.add(32 + 16 * h));
                let qh = vld1q_u8(qh.add(16 * h));
                let q = q.add(128 * n + 16 * h);
                store(q, vorrq_u8(vandq_u8(ql0, low), q6_high(qh, 0)));
                store(q.add(32), vorrq_u8(vandq_u8(ql1, low), q6_high(qh, 2)));
                store(q.add(64), vorrq_u8(vshrq_n_u8::<4>(ql0), q6_high(qh, 4)));
                store(q.add(96), vorrq_u8(vshrq_n_u8::<4>(ql1), q6_high(qh, 6)));
            }
        }
        let sc = &blk[192..208];
        for (u, &s) in u.sc.iter_mut().zip(sc) {
            *u = s as i8 as i32;
        }
        for p in 0..8 {
            u.pairs[p] = pair(sc[2 * p] as i8 as i16, sc[2 * p + 1] as i8 as i16);
        }
        u.d = half(&blk[208..210]);
        u.dmin = 0.0;
    }
}

#[target_feature(enable = "neon")]
unsafe fn unpack_q8_0(row: &[u8], out: &mut [Unpacked32]) {
    for (blk, u) in row.chunks_exact(34).zip(out) {
        u.d = half(blk);
        u.q.copy_from_slice(&blk[2..34]);
    }
}

#[target_feature(enable = "neon")]
unsafe fn unpack_q4_0(row: &[u8], out: &mut [Unpacked32]) {
    let low = vdupq_n_u8(0x0F);
    for (blk, u) in row.chunks_exact(18).zip(out) {
        u.d = half(blk);
        // Low nibbles are elements 0..16, high nibbles 16..32.
        let q = vld1q_u8(blk[2..].as_ptr());
        store(u.q.as_mut_ptr(), vandq_u8(q, low));
        store(u.q.as_mut_ptr().add(16), vshrq_n_u8::<4>(q));
    }
}

/// The first `V` (4, 8 or 16) bytes at `q` in every `V`-byte lane.
#[inline(always)]
unsafe fn broadcast<const V: usize>(q: *const u8) -> int8x16_t {
    match V {
        4 => vreinterpretq_s8_u32(vdupq_n_u32(q.cast::<u32>().read_unaligned())),
        8 => vreinterpretq_s8_u64(vdupq_n_u64(q.cast::<u64>().read_unaligned())),
        _ => vld1q_s8(q.cast()),
    }
}

/// Two vectors of four lanes: tokens 0..4 and 4..8.
type Lanes8<T> = [T; 2];

/// Adds the lanes of each token of a step, so token `t` ends up in lane
/// `t` of the pair.
#[inline(always)]
unsafe fn token_sums<const V: usize>(acc: Lanes8<int32x4_t>) -> Lanes8<int32x4_t> {
    let zero = vdupq_n_s32(0);
    match V {
        4 => acc,
        8 => [vpaddq_s32(acc[0], acc[1]), zero],
        _ => [vpaddq_s32(vpaddq_s32(acc[0], acc[1]), zero), zero],
    }
}

/// Per token, `sum(pairs[p].lo * sums[p][t][0] + pairs[p].hi * sums[p][t][1])`.
#[inline(always)]
unsafe fn pair_dot(sums: &[i16], pairs: &[i32]) -> Lanes8<int32x4_t> {
    let mut acc = [vdupq_n_s32(0); 2];
    for (p, &f) in pairs.iter().enumerate() {
        let s = sums[p * 2 * GROUP..][..2 * GROUP].as_ptr();
        let f = vreinterpret_s16_s32(vdup_n_s32(f));
        for (acc, half) in acc.iter_mut().zip([s, s.add(GROUP)]) {
            let s = vld1q_s16(half);
            let lo = vmull_s16(vget_low_s16(s), f);
            let hi = vmull_s16(vget_high_s16(s), f);
            *acc = vaddq_s32(*acc, vpaddq_s32(lo, hi));
        }
    }
    acc
}

#[inline(always)]
unsafe fn to_f32(v: Lanes8<int32x4_t>) -> Lanes8<float32x4_t> {
    [vcvtq_f32_s32(v[0]), vcvtq_f32_s32(v[1])]
}

#[inline(always)]
unsafe fn scaled(dx: Lanes8<float32x4_t>, d: f32) -> Lanes8<float32x4_t> {
    [vmulq_n_f32(dx[0], d), vmulq_n_f32(dx[1], d)]
}

#[inline(always)]
unsafe fn mul(a: Lanes8<float32x4_t>, b: Lanes8<float32x4_t>) -> Lanes8<float32x4_t> {
    [vmulq_f32(a[0], b[0]), vmulq_f32(a[1], b[1])]
}

/// Eight partial sums per row by block lane, as in the one-row kernels.
type Part<const R: usize> = [[Lanes8<float32x4_t>; 8]; R];

#[inline(always)]
unsafe fn accumulate<const R: usize>(
    part: &mut Part<R>,
    r: usize,
    b: usize,
    v: Lanes8<float32x4_t>,
) {
    let p = &mut part[r][b % 8];
    p[0] = vaddq_f32(p[0], v[0]);
    p[1] = vaddq_f32(p[1], v[1]);
}

/// Adds each row's eight partial sums in the lane order of the one-row
/// kernels: (0 + 4) + (2 + 6), then (1 + 5) + (3 + 7).
#[inline(always)]
unsafe fn reduce<const R: usize>(part: &Part<R>) -> [[f32; GROUP]; R] {
    let mut out = [[0.0; GROUP]; R];
    for (out, p) in out.iter_mut().zip(part) {
        for (h, out) in out.chunks_exact_mut(4).enumerate() {
            let (q0, q1) = (vaddq_f32(p[0][h], p[4][h]), vaddq_f32(p[1][h], p[5][h]));
            let (q2, q3) = (vaddq_f32(p[2][h], p[6][h]), vaddq_f32(p[3][h], p[7][h]));
            let sum = vaddq_f32(vaddq_f32(q0, q2), vaddq_f32(q1, q3));
            vst1q_f32(out.as_mut_ptr(), sum);
        }
    }
    out
}

/// [`tile`] for the width of `x`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn tiles<const R: usize, const MINS: bool>(
    rows: [&[Unpacked]; R],
    x: &Q8KGroup,
) -> [[f32; GROUP]; R] {
    match x.width {
        GROUP => tile::<R, MINS, 4>(rows, x),
        4 => tile::<R, MINS, 8>(rows, x),
        _ => tile::<R, MINS, 16>(rows, x),
    }
}

/// `R` rows times the tokens of `x`, `V` values of each per step. With
/// `MINS` (Q4_K and Q5_K) blocks subtract a min per 32 values; without,
/// weights are stored as `q + 32` (Q6_K), which comes out of the integer
/// sum through the 16-value sums.
#[target_feature(enable = "neon,dotprod")]
unsafe fn tile<const R: usize, const MINS: bool, const V: usize>(
    rows: [&[Unpacked]; R],
    x: &Q8KGroup,
) -> [[f32; GROUP]; R] {
    let blocks = x.blocks();
    for row in rows {
        assert_eq!(row.len(), blocks);
    }
    let width = 32 / V;
    let block_bytes = 256 * width;
    // Steps of one 16-value scale group.
    let steps = 16 / V;
    let zero = vdupq_n_s32(0);
    let mut part: Part<R> = [[[vdupq_n_f32(0.0); 2]; 8]; R];
    for (b, dx) in x.d.chunks_exact(GROUP).enumerate() {
        let qs = x.qs[b * block_bytes..][..block_bytes].as_ptr();
        let mut isum = [[zero; 2]; R];
        for g in 0..16 {
            // The activations of one scale group stay in registers for all
            // rows; each row's weights are broadcast `V` bytes at a time.
            let mut y = [[vdupq_n_s8(0); 2]; 4];
            for (s, y) in y[..steps].iter_mut().enumerate() {
                let p = qs.add(32 * (g * steps + s));
                *y = [vld1q_s8(p), vld1q_s8(p.add(16))];
            }
            for (row, isum) in rows.iter().zip(isum.iter_mut()) {
                let u = &row[b];
                let mut acc = [zero; 2];
                for (s, y) in y[..steps].iter().enumerate() {
                    let w = broadcast::<V>(u.q.as_ptr().add(16 * g + V * s));
                    acc[0] = sdot(acc[0], w, y[0]);
                    acc[1] = sdot(acc[1], w, y[1]);
                }
                isum[0] = vmlaq_n_s32(isum[0], acc[0], u.sc[g]);
                isum[1] = vmlaq_n_s32(isum[1], acc[1], u.sc[g]);
            }
        }
        let dx = [vld1q_f32(dx.as_ptr()), vld1q_f32(dx.as_ptr().add(4))];
        for (r, row) in rows.iter().enumerate() {
            let u = &row[b];
            let isum = token_sums::<V>(isum[r]);
            let dd = scaled(dx, u.d);
            let v = if MINS {
                let sums32 = &x.sums32[b * 4 * 2 * GROUP..][..4 * 2 * GROUP];
                let imin = to_f32(pair_dot(sums32, &u.pairs[..4]));
                let dm = scaled(dx, u.dmin);
                let v = mul(dd, to_f32(isum));
                [
                    vfmsq_f32(v[0], dm[0], imin[0]),
                    vfmsq_f32(v[1], dm[1], imin[1]),
                ]
            } else {
                let sums16 = &x.sums16[b * 8 * 2 * GROUP..][..8 * 2 * GROUP];
                let bias = pair_dot(sums16, &u.pairs);
                let total = [
                    vsubq_s32(isum[0], vshlq_n_s32::<5>(bias[0])),
                    vsubq_s32(isum[1], vshlq_n_s32::<5>(bias[1])),
                ];
                mul(dd, to_f32(total))
            };
            accumulate(&mut part, r, b, v);
        }
    }
    reduce(&part)
}

/// [`tile32`] for the width of `x`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn tiles32<const R: usize, const OFFSET: bool>(
    rows: [&[Unpacked32]; R],
    x: &Q8Group,
) -> [[f32; GROUP]; R] {
    match x.width {
        GROUP => tile32::<R, OFFSET, 4>(rows, x),
        4 => tile32::<R, OFFSET, 8>(rows, x),
        _ => tile32::<R, OFFSET, 16>(rows, x),
    }
}

/// `R` rows of Q8_0 (signed weights in `q`) or, with `OFFSET`, Q4_0
/// (codes stored as `q + 8`, which leaves through the block sums) times
/// the eight tokens of `x`, with the f32 steps of the one-row kernels.
#[target_feature(enable = "neon,dotprod")]
unsafe fn tile32<const R: usize, const OFFSET: bool, const V: usize>(
    rows: [&[Unpacked32]; R],
    x: &Q8Group,
) -> [[f32; GROUP]; R] {
    let blocks = x.blocks();
    for row in rows {
        assert_eq!(row.len(), blocks);
    }
    let width = 32 / V;
    let zero = vdupq_n_s32(0);
    let mut part: Part<R> = [[[vdupq_n_f32(0.0); 2]; 8]; R];
    for (b, dx) in x.d.chunks_exact(GROUP).enumerate() {
        let qs = x.qs[b * 32 * width..][..32 * width].as_ptr();
        let mut y = [[vdupq_n_s8(0); 2]; 8];
        for (k, y) in y[..width].iter_mut().enumerate() {
            let p = qs.add(32 * k);
            *y = [vld1q_s8(p), vld1q_s8(p.add(16))];
        }
        let dx = [vld1q_f32(dx.as_ptr()), vld1q_f32(dx.as_ptr().add(4))];
        let sums = &x.sums[b * GROUP..][..GROUP];
        let sums = [vld1q_s32(sums.as_ptr()), vld1q_s32(sums.as_ptr().add(4))];
        for (r, row) in rows.iter().enumerate() {
            let u = &row[b];
            let mut acc = [zero; 2];
            for (k, y) in y[..width].iter().enumerate() {
                let w = broadcast::<V>(u.q.as_ptr().add(V * k));
                acc[0] = sdot(acc[0], w, y[0]);
                acc[1] = sdot(acc[1], w, y[1]);
            }
            let mut isum = token_sums::<V>(acc);
            if OFFSET {
                isum[0] = vsubq_s32(isum[0], vmulq_n_s32(sums[0], 8));
                isum[1] = vsubq_s32(isum[1], vmulq_n_s32(sums[1], 8));
            }
            let v = mul(scaled(dx, u.d), to_f32(isum));
            accumulate(&mut part, r, b, v);
        }
    }
    reduce(&part)
}
