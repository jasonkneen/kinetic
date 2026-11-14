//! NEON versions of the one-row kernels for AArch64 CPUs with the SDOT
//! instruction: every Apple Silicon chip, and Arm cores from Cortex-A75 on.
//!
//! Each block's integer sums are exact, and the f32 fold that follows does
//! the operations of the AVX2 kernels in the same order, lane for lane, so
//! a value is the same on both architectures.

use std::arch::aarch64::*;
use std::arch::asm;

use super::{half, mins_term, scales_mins, DotFn, DotNFn};
use crate::q8::Q8;
use crate::q8k::Q8K;
use kinetic_quant::TensorType;

pub fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

pub fn kernel(ty: TensorType) -> Option<DotFn<Q8K>> {
    match ty {
        TensorType::Q4K => Some(q4_k),
        TensorType::Q5K => Some(q5_k),
        TensorType::Q6K => Some(q6_k),
        _ => None,
    }
}

/// The kernel for `ty` over `N` activation vectors at once.
pub fn kernel_n<const N: usize>(ty: TensorType) -> Option<DotNFn<Q8K, N>> {
    // SAFETY (each closure): only reachable after `available()` confirmed
    // the target features the implementations are compiled for.
    match ty {
        TensorType::Q4K => Some(|row, xs| unsafe { q4_k_impl::<N>(row, xs) }),
        TensorType::Q5K => Some(|row, xs| unsafe { q5_k_impl::<N>(row, xs) }),
        TensorType::Q6K => Some(|row, xs| unsafe { q6_k_impl::<N>(row, xs) }),
        _ => None,
    }
}

pub fn kernel32(ty: TensorType) -> Option<DotFn<Q8>> {
    match ty {
        TensorType::Q8_0 => Some(q8_0),
        TensorType::Q4_0 => Some(q4_0),
        _ => None,
    }
}

/// The kernel for a 32-value format `ty` over `N` activation vectors.
pub fn kernel32_n<const N: usize>(ty: TensorType) -> Option<DotNFn<Q8, N>> {
    // SAFETY (each closure): as in `kernel_n`.
    match ty {
        TensorType::Q8_0 => Some(|row, xs| unsafe { q8_0_impl::<N>(row, xs) }),
        TensorType::Q4_0 => Some(|row, xs| unsafe { q4_0_impl::<N>(row, xs) }),
        _ => None,
    }
}

// The safe wrappers are only handed out by `kernel`, which callers reach
// through `kdot::kernel` after `available()` has confirmed the features.

fn q4_k(row: &[u8], x: &Q8K) -> f32 {
    unsafe { q4_k_impl(row, [x])[0] }
}

fn q5_k(row: &[u8], x: &Q8K) -> f32 {
    unsafe { q5_k_impl(row, [x])[0] }
}

fn q6_k(row: &[u8], x: &Q8K) -> f32 {
    unsafe { q6_k_impl(row, [x])[0] }
}

fn q8_0(row: &[u8], x: &Q8) -> f32 {
    unsafe { q8_0_impl(row, [x])[0] }
}

fn q4_0(row: &[u8], x: &Q8) -> f32 {
    unsafe { q4_0_impl(row, [x])[0] }
}

/// `acc + a . b` over each group of four lanes: the SDOT instruction, whose
/// intrinsic the standard library still keeps behind a nightly feature.
#[inline(always)]
pub(super) unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut out = acc;
    asm!(
        "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
        inout(vreg) out,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack)
    );
    out
}

/// `a . b` over 32 bytes, as four partial lanes.
#[inline(always)]
unsafe fn dot32(a: [int8x16_t; 2], b: *const i8) -> int32x4_t {
    let acc = sdot(vdupq_n_s32(0), a[0], vld1q_s8(b));
    sdot(acc, a[1], vld1q_s8(b.add(16)))
}

/// Unsigned bytes below 128 as signed, which SDOT takes.
#[inline(always)]
unsafe fn signed(v: uint8x16_t) -> int8x16_t {
    vreinterpretq_s8_u8(v)
}

/// Eight running sums, block `b` into lane `b % 8` in block order, added
/// at the end the way the AVX2 `hsum` adds its lanes.
struct Lanes([f32; 8]);

impl Lanes {
    const ZERO: Lanes = Lanes([0.0; 8]);

    #[inline(always)]
    fn add(&mut self, b: usize, v: f32) {
        self.0[b % 8] += v;
    }

    #[inline(always)]
    fn total(&self) -> f32 {
        let l = &self.0;
        let s = [l[0] + l[4], l[1] + l[5], l[2] + l[6], l[3] + l[7]];
        (s[0] + s[2]) + (s[1] + s[3])
    }
}

/// Block `b`'s `(dx * d) * isum - (dx * dmin) * imin`, the second term
/// fused, into its lane.
#[inline(always)]
fn fold_mins(part: &mut Lanes, b: usize, dx: f32, d: f32, dmin: f32, isum: i32, imin: i32) {
    let v = (dx * d) * isum as f32;
    part.add(b, (-(dx * dmin)).mul_add(imin as f32, v));
}

/// Block `b`'s `(dx * d) * isum` into its lane.
#[inline(always)]
fn fold(part: &mut Lanes, b: usize, dx: f32, d: f32, isum: i32) {
    part.add(b, (dx * d) * isum as f32);
}

// Each kernel takes N activation vectors so one unpacking of a weight block
// serves N tokens. N = 1 is decode; short batches use up to N = 4.

#[target_feature(enable = "neon,dotprod")]
unsafe fn q4_k_impl<const N: usize>(row: &[u8], xs: [&Q8K; N]) -> [f32; N] {
    let low = vdupq_n_u8(0x0F);
    let mut part = [Lanes::ZERO; N];
    for (b, blk) in row.chunks_exact(144).enumerate() {
        let (d, dmin) = (half(&blk[0..2]), half(&blk[2..4]));
        let (sc, m) = scales_mins(&blk[4..16]);
        let qs = blk[16..].as_ptr();
        let mut sumi = [vdupq_n_s32(0); N];
        for j in 0..4 {
            let q0 = vld1q_u8(qs.add(32 * j));
            let q1 = vld1q_u8(qs.add(32 * j + 16));
            let lo = [signed(vandq_u8(q0, low)), signed(vandq_u8(q1, low))];
            let hi = [signed(vshrq_n_u8::<4>(q0)), signed(vshrq_n_u8::<4>(q1))];
            for i in 0..N {
                let y = xs[i].qs.as_ptr().add(b * 256 + 64 * j);
                sumi[i] = vmlaq_n_s32(sumi[i], dot32(lo, y), sc[2 * j] as i32);
                sumi[i] = vmlaq_n_s32(sumi[i], dot32(hi, y.add(32)), sc[2 * j + 1] as i32);
            }
        }
        for i in 0..N {
            let imin = mins_term(&m, xs[i].block(b).2);
            fold_mins(
                &mut part[i],
                b,
                xs[i].d[b],
                d,
                dmin,
                vaddvq_s32(sumi[i]),
                imin,
            );
        }
    }
    std::array::from_fn(|i| part[i].total())
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn q5_k_impl<const N: usize>(row: &[u8], xs: [&Q8K; N]) -> [f32; N] {
    let low = vdupq_n_u8(0x0F);
    let bit4 = vdupq_n_u8(0x10);
    let mut part = [Lanes::ZERO; N];
    for (b, blk) in row.chunks_exact(176).enumerate() {
        let (d, dmin) = (half(&blk[0..2]), half(&blk[2..4]));
        let (sc, m) = scales_mins(&blk[4..16]);
        // Bit t of each qh byte is the fifth bit of sub-block t; a shift
        // by (4 - t), to the right when negative, brings it to position 4.
        let qh = [vld1q_u8(blk[16..].as_ptr()), vld1q_u8(blk[32..].as_ptr())];
        let fifth =
            |qh: uint8x16_t, t: usize| vandq_u8(vshlq_u8(qh, vdupq_n_s8(4 - t as i8)), bit4);
        let qs = blk[48..].as_ptr();
        let mut sumi = [vdupq_n_s32(0); N];
        for j in 0..4 {
            let q0 = vld1q_u8(qs.add(32 * j));
            let q1 = vld1q_u8(qs.add(32 * j + 16));
            let lo = [
                signed(vorrq_u8(vandq_u8(q0, low), fifth(qh[0], 2 * j))),
                signed(vorrq_u8(vandq_u8(q1, low), fifth(qh[1], 2 * j))),
            ];
            let hi = [
                signed(vorrq_u8(vshrq_n_u8::<4>(q0), fifth(qh[0], 2 * j + 1))),
                signed(vorrq_u8(vshrq_n_u8::<4>(q1), fifth(qh[1], 2 * j + 1))),
            ];
            for i in 0..N {
                let y = xs[i].qs.as_ptr().add(b * 256 + 64 * j);
                sumi[i] = vmlaq_n_s32(sumi[i], dot32(lo, y), sc[2 * j] as i32);
                sumi[i] = vmlaq_n_s32(sumi[i], dot32(hi, y.add(32)), sc[2 * j + 1] as i32);
            }
        }
        for i in 0..N {
            let imin = mins_term(&m, xs[i].block(b).2);
            fold_mins(
                &mut part[i],
                b,
                xs[i].d[b],
                d,
                dmin,
                vaddvq_s32(sumi[i]),
                imin,
            );
        }
    }
    std::array::from_fn(|i| part[i].total())
}

/// Bits `shift..shift + 2` of each qh byte, moved to bits 4..6: one shift
/// by `4 - shift`, to the right when negative, then a mask.
#[inline(always)]
unsafe fn q6_high(qh: uint8x16_t, shift: i8) -> uint8x16_t {
    vandq_u8(vshlq_u8(qh, vdupq_n_s8(4 - shift)), vdupq_n_u8(0x30))
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn q6_k_impl<const N: usize>(row: &[u8], xs: [&Q8K; N]) -> [f32; N] {
    let low = vdupq_n_u8(0x0F);
    let mut part = [Lanes::ZERO; N];
    for (b, blk) in row.chunks_exact(210).enumerate() {
        let sc = &blk[192..208];
        let d = half(&blk[208..210]);
        let mut sumi = [vdupq_n_s32(0); N];
        for n in 0..2 {
            let ql = blk[64 * n..].as_ptr();
            let qh = blk[128 + 32 * n..].as_ptr();
            // Weights are stored as q + 32; the 32 comes back out through
            // the per-16 sums below, which line up one-to-one with the
            // scales. Each group is 32 elements, two 16-byte halves with
            // their own scales.
            for h in 0..2 {
                let ql0 = vld1q_u8(ql.add(16 * h));
                let ql1 = vld1q_u8(ql.add(32 + 16 * h));
                let qh = vld1q_u8(qh.add(16 * h));
                let groups = [
                    vorrq_u8(vandq_u8(ql0, low), q6_high(qh, 0)),
                    vorrq_u8(vandq_u8(ql1, low), q6_high(qh, 2)),
                    vorrq_u8(vshrq_n_u8::<4>(ql0), q6_high(qh, 4)),
                    vorrq_u8(vshrq_n_u8::<4>(ql1), q6_high(qh, 6)),
                ];
                for (g, q) in groups.into_iter().enumerate() {
                    let e = 128 * n + 32 * g + 16 * h;
                    let s = sc[e / 16] as i8 as i32;
                    for i in 0..N {
                        let y = vld1q_s8(xs[i].qs.as_ptr().add(b * 256 + e));
                        sumi[i] = vmlaq_n_s32(sumi[i], sdot(vdupq_n_s32(0), signed(q), y), s);
                    }
                }
            }
        }
        for i in 0..N {
            let bsums = xs[i].block(b).2;
            let bias: i32 = (0..16).map(|k| sc[k] as i8 as i32 * bsums[k] as i32).sum();
            let total = vaddvq_s32(sumi[i]) - (bias << 5);
            fold(&mut part[i], b, xs[i].d[b], d, total);
        }
    }
    std::array::from_fn(|i| part[i].total())
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn q8_0_impl<const N: usize>(row: &[u8], xs: [&Q8; N]) -> [f32; N] {
    let mut part = [Lanes::ZERO; N];
    for (k, blk) in row.chunks_exact(34).enumerate() {
        let dw = half(blk);
        let w = [
            vld1q_s8(blk[2..].as_ptr().cast()),
            vld1q_s8(blk[18..].as_ptr().cast()),
        ];
        for (part, x) in part.iter_mut().zip(xs) {
            let isum = vaddvq_s32(dot32(w, x.qs.as_ptr().add(32 * k)));
            fold(part, k, x.d[k], dw, isum);
        }
    }
    std::array::from_fn(|i| part[i].total())
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn q4_0_impl<const N: usize>(row: &[u8], xs: [&Q8; N]) -> [f32; N] {
    let low = vdupq_n_u8(0x0F);
    let mut part = [Lanes::ZERO; N];
    for (k, blk) in row.chunks_exact(18).enumerate() {
        let dw = half(blk);
        // Low nibbles are elements 0..16, high nibbles 16..32.
        let q = vld1q_u8(blk[2..].as_ptr());
        let w = [signed(vandq_u8(q, low)), signed(vshrq_n_u8::<4>(q))];
        for (part, x) in part.iter_mut().zip(xs) {
            // weights are stored as q + 8; the 8 leaves through the block sums
            let isum = vaddvq_s32(dot32(w, x.qs.as_ptr().add(32 * k))) - 8 * x.sums[k] as i32;
            fold(part, k, x.d[k], dw, isum);
        }
    }
    std::array::from_fn(|i| part[i].total())
}
