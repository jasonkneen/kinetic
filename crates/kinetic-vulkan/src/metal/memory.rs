use std::marker::PhantomData;
use std::mem::size_of;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::commands::Commands;
use crate::device::Device;
use crate::error::{Error, Result};

mod sealed {
    pub trait Sealed {}
}

/// Plain values that can be copied to and from the device byte for byte.
pub trait DeviceCopy: Copy + Default + sealed::Sealed {}

macro_rules! device_copy {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl DeviceCopy for $t {}
    )*};
}

device_copy!(u8, i8, u16, u32, f32);

/// A Metal buffer in shared memory, which the host and the GPU both
/// address. Recorded commands hold an `Rc` to each allocation they use, so
/// it outlives their execution.
///
/// The device touches a buffer only while commands run, and running them
/// waits for them, so host access never races with the device.
pub(crate) struct Allocation {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl Allocation {
    /// `bytes` zeroed bytes.
    fn new(device: &Rc<Device>, bytes: usize) -> Result<Allocation> {
        // Kernels read whole 16-byte words; an empty buffer is not allowed.
        let length = bytes.max(4).next_multiple_of(16);
        let buffer = device
            .raw
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| Error::Metal {
                call: "newBufferWithLength",
                why: format!("no room for {length} bytes"),
            })?;
        let alloc = Allocation { buffer };
        // SAFETY: the buffer holds `length` bytes and nothing else uses it yet.
        unsafe { std::ptr::write_bytes(alloc.ptr(), 0, length) };
        Ok(alloc)
    }

    fn ptr(&self) -> *mut u8 {
        self.buffer.contents().as_ptr().cast()
    }
}

/// An array of `T` in device memory, bindable as a storage buffer.
pub struct DeviceBuffer<T: DeviceCopy> {
    alloc: Rc<Allocation>,
    len: usize,
    _elements: PhantomData<T>,
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Allocates `len` zeroed elements.
    pub fn new(device: &Rc<Device>, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .filter(|&b| b as u64 <= device.info().max_buffer_bytes)
            .ok_or_else(|| {
                Error::Shape(format!(
                    "buffer of {len} elements exceeds the device's buffer limit"
                ))
            })?;
        Ok(DeviceBuffer {
            alloc: Rc::new(Allocation::new(device, bytes)?),
            len,
            _elements: PhantomData,
        })
    }

    /// Sets every element to zero.
    pub fn zero(&mut self) -> Result<()> {
        // SAFETY: the buffer holds `len` elements; see `Allocation` on races.
        unsafe { std::ptr::write_bytes(self.alloc.ptr().cast::<T>(), 0, self.len) };
        Ok(())
    }

    pub fn from_slice(device: &Rc<Device>, data: &[T]) -> Result<Self> {
        let mut buf = Self::new(device, data.len())?;
        buf.copy_from(data)?;
        Ok(buf)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn allocation(&self) -> &Rc<Allocation> {
        &self.alloc
    }

    /// Copies `src` into the buffer.
    pub fn copy_from(&mut self, src: &[T]) -> Result<()> {
        self.check_len(src.len())?;
        self.copy_range_from(0, src)
    }

    /// Copies the buffer into `dst`.
    pub fn copy_to(&self, dst: &mut [T]) -> Result<()> {
        self.check_len(dst.len())?;
        self.copy_range_to(0, dst)
    }

    fn check_len(&self, other: usize) -> Result<()> {
        if other == self.len {
            Ok(())
        } else {
            Err(Error::Shape(format!(
                "copy of {other} elements into or out of a buffer of {}",
                self.len
            )))
        }
    }

    /// Copies `src` into the buffer from element `at`.
    pub fn copy_range_from(&mut self, at: usize, src: &[T]) -> Result<()> {
        check_range("copy", at, src.len(), self.len)?;
        // SAFETY: the range was checked; see `Allocation` on races.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                self.alloc.ptr().cast::<T>().add(at),
                src.len(),
            )
        };
        Ok(())
    }

    /// Copies `dst.len()` elements from element `at` into `dst`.
    pub fn copy_range_to(&self, at: usize, dst: &mut [T]) -> Result<()> {
        check_range("copy", at, dst.len(), self.len)?;
        // SAFETY: the range was checked; see `Allocation` on races.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.alloc.ptr().cast::<T>().add(at),
                dst.as_mut_ptr(),
                dst.len(),
            )
        };
        Ok(())
    }

    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut out = vec![T::default(); self.len];
        self.copy_to(&mut out)?;
        Ok(out)
    }
}

/// An array of `T` the host reads and writes in place; recorded commands
/// copy between it and device buffers.
pub struct HostBuffer<T: DeviceCopy> {
    alloc: Rc<Allocation>,
    ptr: *mut T,
    len: usize,
}

impl<T: DeviceCopy> HostBuffer<T> {
    /// Allocates `len` zeroed elements.
    pub fn new(device: &Rc<Device>, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::Shape(format!("host buffer of {len} elements")))?;
        let alloc = Allocation::new(device, bytes)?;
        Ok(HostBuffer {
            ptr: alloc.ptr().cast(),
            alloc: Rc::new(alloc),
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the buffer holds `len` initialised elements and lives as
        // long as `self`; see `Allocation` on races.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`, and `&mut self` makes the access unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub(crate) fn allocation(&self) -> &Rc<Allocation> {
        &self.alloc
    }
}

fn check_range(what: &str, at: usize, len: usize, size: usize) -> Result<()> {
    match at.checked_add(len) {
        Some(end) if end <= size => Ok(()),
        _ => Err(Error::Shape(format!(
            "{what} of {len} elements at {at} past a buffer of {size}"
        ))),
    }
}

impl Commands {
    /// Records a copy of `len` elements from `src` at `src_at` to `dst` at
    /// `dst_at`.
    pub fn upload<T: DeviceCopy>(
        &mut self,
        src: &HostBuffer<T>,
        src_at: usize,
        dst: &DeviceBuffer<T>,
        dst_at: usize,
        len: usize,
    ) -> Result<()> {
        check_range("upload", src_at, len, src.len())?;
        check_range("upload", dst_at, len, dst.len())?;
        let t = size_of::<T>();
        // SAFETY: both ranges were checked above.
        unsafe {
            self.copy(
                src.allocation(),
                src_at * t,
                dst.allocation(),
                dst_at * t,
                len * t,
            )
        };
        Ok(())
    }

    /// Records a copy of `len` elements from `src` at `src_at` to `dst` at
    /// `dst_at`.
    pub fn download<T: DeviceCopy>(
        &mut self,
        src: &DeviceBuffer<T>,
        src_at: usize,
        dst: &HostBuffer<T>,
        dst_at: usize,
        len: usize,
    ) -> Result<()> {
        check_range("download", src_at, len, src.len())?;
        check_range("download", dst_at, len, dst.len())?;
        let t = size_of::<T>();
        // SAFETY: both ranges were checked above.
        unsafe {
            self.copy(
                src.allocation(),
                src_at * t,
                dst.allocation(),
                dst_at * t,
                len * t,
            )
        };
        Ok(())
    }
}
