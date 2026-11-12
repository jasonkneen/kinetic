//! This computer: operating system, CPU, memory, disks and GPUs.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(windows)]
mod windows;

mod gpu;

#[cfg(target_os = "linux")]
use linux as os;
#[cfg(target_os = "macos")]
use macos as os;
#[cfg(windows)]
use windows as os;

use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

pub use gpu::{cuda_free, gpus, Gpu};

/// The CPU and memory.
pub struct Machine {
    pub os: String,
    pub cpu: String,
    pub threads: usize,
    /// The instruction sets Kinetic's CPU kernels use on this
    /// architecture, each with whether this CPU has it.
    pub isa: Vec<(&'static str, bool)>,
    pub ram_total: u64,
    pub ram_available: u64,
}

pub fn machine() -> Machine {
    let (ram_total, ram_available) = os::memory().unwrap_or((0, 0));
    Machine {
        os: os::name(),
        cpu: cpu_name().unwrap_or_else(|| std::env::consts::ARCH.to_owned()),
        threads: default_threads()
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())),
        isa: features(),
        ram_total,
        ram_available,
    }
}

/// How many threads the CPU kernels should use when the user says nothing:
/// the performance cores on macOS, where the efficiency cores would hold
/// every parallel step back; `None` elsewhere, for every logical CPU.
pub fn default_threads() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        macos::performance_cores().filter(|&n| n > 0)
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// Bytes free for this user on the disk holding `path`.
pub fn disk_free(path: &Path) -> Option<u64> {
    let mut dir = path;
    // the folder may not exist yet; ask about the nearest one that does
    while !dir.exists() {
        dir = dir.parent()?;
    }
    os::disk_free(dir)
}

impl Machine {
    /// Whether the CPU has everything the fast kernels need.
    pub fn fast_kernels(&self) -> bool {
        self.isa.iter().all(|&(_, has)| has)
    }

    /// "AVX2 yes, FMA yes, F16C NO".
    pub fn isa_summary(&self) -> String {
        self.isa
            .iter()
            .map(|&(name, has)| format!("{name} {}", if has { "yes" } else { "NO" }))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The instruction sets the fast kernels need, for a message that
    /// says what is missing.
    pub fn isa_needed(&self) -> String {
        let names: Vec<_> = self.isa.iter().map(|&(name, _)| name).collect();
        match names.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
            Some((last, _)) => (*last).to_owned(),
            None => String::new(),
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn features() -> Vec<(&'static str, bool)> {
    vec![
        ("AVX2", is_x86_feature_detected!("avx2")),
        ("FMA", is_x86_feature_detected!("fma")),
        ("F16C", is_x86_feature_detected!("f16c")),
    ]
}

// NEON is part of every AArch64 CPU; the integer dot product (SDOT/UDOT)
// and half-precision arithmetic are what the quantized kernels want.
#[cfg(target_arch = "aarch64")]
fn features() -> Vec<(&'static str, bool)> {
    vec![
        ("NEON", true),
        (
            "DotProd",
            std::arch::is_aarch64_feature_detected!("dotprod"),
        ),
        ("FP16", std::arch::is_aarch64_feature_detected!("fp16")),
    ]
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn features() -> Vec<(&'static str, bool)> {
    Vec::new()
}

/// The brand string the CPU reports, from cpuid leaves 0x80000002..4.
#[cfg(target_arch = "x86_64")]
// __cpuid is a safe fn on toolchains newer than the 1.91 MSRV
#[allow(unused_unsafe)]
fn cpu_name() -> Option<String> {
    use std::arch::x86_64::__cpuid;
    // SAFETY: cpuid exists on every x86-64 CPU; the extended leaves are
    // read only after leaf 0x80000000 says they exist.
    let max = unsafe { __cpuid(0x8000_0000) }.eax;
    if max < 0x8000_0004 {
        return None;
    }
    let mut bytes = Vec::with_capacity(48);
    for leaf in 0x8000_0002..=0x8000_0004u32 {
        // SAFETY: as above, the leaf is within the reported range.
        let r = unsafe { __cpuid(leaf) };
        for word in [r.eax, r.ebx, r.ecx, r.edx] {
            bytes.extend(word.to_le_bytes());
        }
    }
    let name = String::from_utf8_lossy(&bytes);
    let name = name.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    (!name.is_empty()).then(|| name.split_whitespace().collect::<Vec<_>>().join(" "))
}

#[cfg(all(not(target_arch = "x86_64"), target_os = "macos"))]
fn cpu_name() -> Option<String> {
    macos::sysctl_string(c"machdep.cpu.brand_string")
}

#[cfg(all(not(target_arch = "x86_64"), not(target_os = "macos")))]
fn cpu_name() -> Option<String> {
    None
}

/// Read bandwidth of main memory in bytes per second, measured by summing
/// a buffer larger than any cache on every thread, best of a few passes.
pub fn memory_bandwidth() -> f64 {
    // the speed limit; everything else in kinetic is about not paying it twice
    const BYTES: usize = 256 << 20;
    let buffer = vec![1u64; BYTES / 8];
    let mut best = f64::MAX;
    for _ in 0..4 {
        let start = Instant::now();
        let sum: u64 = buffer
            .par_chunks(1 << 16)
            .map(|c| c.iter().sum::<u64>())
            .sum();
        std::hint::black_box(sum);
        best = best.min(start.elapsed().as_secs_f64());
    }
    BYTES as f64 / best
}
