use std::ffi::{CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

// libc deprecates its binding in favour of the mach2 crate; the symbol
// itself is stable, so it is declared here rather than taking a dependency.
extern "C" {
    fn mach_host_self() -> libc::mach_port_t;
}

/// The string value of a sysctl, such as `kern.osproductversion`.
pub fn sysctl_string(name: &CStr) -> Option<String> {
    let mut len = 0usize;
    // SAFETY: a null buffer asks only for the length.
    if unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    let mut buf = vec![0u8; len];
    // SAFETY: `buf` holds `len` bytes, as reported by the first call.
    if unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    buf.truncate(len);
    let s = String::from_utf8_lossy(&buf);
    let s = s.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    (!s.is_empty()).then(|| s.to_owned())
}

/// The integer value of a sysctl, such as `hw.memsize`; 32-bit values,
/// such as CPU counts, are widened.
fn sysctl_u64(name: &CStr) -> Option<u64> {
    let mut value = 0u64;
    let mut len = size_of::<u64>();
    // SAFETY: `value` is writable and `len` is its size.
    let ok = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    match (ok, len) {
        (0, 8) => Some(value),
        (0, 4) => Some(u64::from(value as u32)),
        _ => None,
    }
}

/// The logical CPUs of the highest performance level (`hw.perflevel0`),
/// the performance cores on Apple Silicon.
pub fn performance_cores() -> Option<usize> {
    sysctl_u64(c"hw.perflevel0.logicalcpu").map(|n| n as usize)
}

/// Total physical memory, and what a new process could take without
/// pushing anything out: free, inactive and speculative pages, the same
/// sum Activity Monitor's "Cached Files" counts as available.
pub fn memory() -> Option<(u64, u64)> {
    let total = sysctl_u64(c"hw.memsize")?;
    // SAFETY: vm_statistics64 is plain data, valid when zeroed.
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: `stats` has room for `count` integers, the size the flavor
    // needs; mach_host_self takes no arguments.
    let ok = unsafe {
        libc::host_statistics64(
            mach_host_self(),
            libc::HOST_VM_INFO64,
            (&mut stats as *mut libc::vm_statistics64).cast(),
            &mut count,
        )
    };
    if ok != libc::KERN_SUCCESS {
        return None;
    }
    // SAFETY: no arguments; returns the VM page size.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page = u64::try_from(page).ok()?;
    let pages = u64::from(stats.free_count)
        + u64::from(stats.inactive_count)
        + u64::from(stats.speculative_count);
    Some((total, pages * page))
}

pub fn disk_free(dir: &Path) -> Option<u64> {
    let path = CString::new(dir.as_os_str().as_bytes()).ok()?;
    // SAFETY: statvfs is plain data, valid when zeroed.
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is NUL-terminated and `s` is writable.
    if unsafe { libc::statvfs(path.as_ptr(), &mut s) } != 0 {
        return None;
    }
    // f_frsize is a c_ulong; f_bavail an fsblkcnt_t, u32 on macOS
    Some(u64::from(s.f_bavail) * s.f_frsize)
}

/// "macOS 26.1 (build 25B74)".
pub fn name() -> String {
    let version = sysctl_string(c"kern.osproductversion");
    let build = sysctl_string(c"kern.osversion");
    match (version, build) {
        (Some(v), Some(b)) => format!("macOS {v} (build {b})"),
        (Some(v), None) => format!("macOS {v}"),
        _ => "macOS".into(),
    }
}
