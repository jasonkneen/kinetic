use std::ffi::c_void;
use std::ptr::NonNull;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLSize,
};

use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::Allocation;
use crate::pipeline::Kernel;

/// Sizes naga's Metal shaders take for their runtime arrays. They only
/// bound checks the kernels do not ask for, so zeros serve.
const ARRAY_SIZES: [u32; 16] = [0; 16];

/// One recorded command.
enum Op {
    Dispatch {
        kernel: Rc<Kernel>,
        buffers: Vec<Rc<Allocation>>,
        push: Vec<u32>,
        groups: [u32; 3],
    },
    Copy {
        src: Rc<Allocation>,
        src_offset: usize,
        dst: Rc<Allocation>,
        dst_offset: usize,
        bytes: usize,
    },
}

/// Kernel launches recorded for one submission.
///
/// Like a CUDA stream, every command waits for the one before it, so a
/// kernel always sees the results of earlier ones: Metal runs the
/// dispatches of an encoder in order. Nothing runs until
/// [`Commands::submit`], which waits for completion.
pub struct Commands {
    device: Rc<Device>,
    ops: Vec<Op>,
    prompt: bool,
}

/// A timestamp recorded in a [`Commands`] list. Metal times command
/// buffers, not points inside them, so none can be recorded.
#[derive(Clone, Copy, Debug)]
pub struct Timestamp(u32);

/// The timestamps of a finished command list.
pub struct Timings {
    ticks: Vec<u64>,
}

impl Timings {
    /// Device time from `start` to `end`.
    pub fn seconds(&self, start: Timestamp, end: Timestamp) -> f64 {
        self.ticks[end.0 as usize].wrapping_sub(self.ticks[start.0 as usize]) as f64
    }
}

impl Commands {
    pub fn new(device: &Rc<Device>) -> Result<Commands> {
        Ok(Commands {
            device: Rc::clone(device),
            ops: Vec::new(),
            prompt: false,
        })
    }

    /// Says whether the commands recorded from here on run a batch of
    /// only prompt tokens, whose results nobody compares with a token's
    /// alone.
    /// The matrix products of such a batch then take the SIMD-group matrix
    /// kernel, which rounds differently from the GEMV kernels.
    pub fn set_prompt(&mut self, prompt: bool) {
        self.prompt = prompt;
    }

    /// See [`Commands::set_prompt`].
    pub(crate) fn prompt(&self) -> bool {
        self.prompt
    }

    /// Records the device time once all earlier commands are done.
    pub fn timestamp(&mut self) -> Result<Timestamp> {
        Err(Error::DeviceUnsupported(
            "Metal has no timestamps inside a command buffer".into(),
        ))
    }

    /// Records the commands `f` records as a group that may run side by
    /// side. Metal is told nothing of it and runs them in order.
    pub fn independent<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        f(self)
    }

    /// Ends the recording, keeping the commands to run as often as wanted.
    pub fn finish(self) -> Result<Graph> {
        Ok(Graph {
            device: self.device,
            ops: self.ops,
        })
    }

    /// Runs the recorded commands and waits for them.
    pub fn submit(self) -> Result<Timings> {
        run(&self.device, &self.ops)?;
        Ok(Timings { ticks: Vec::new() })
    }

    /// Records `kernel` over `groups` workgroups with `buffers` at bindings
    /// 0, 1, ... and `push` as its push constants.
    ///
    /// # Safety
    ///
    /// The buffers and constants must be what the kernel's shader declares,
    /// and the shader must stay inside the buffers for these constants.
    pub(crate) unsafe fn dispatch(
        &mut self,
        kernel: &Rc<Kernel>,
        buffers: &[&Rc<Allocation>],
        push: &[u32],
        groups: [u32; 3],
    ) {
        debug_assert_eq!(buffers.len(), kernel.bindings);
        debug_assert_eq!(push.len(), kernel.push_words);
        self.ops.push(Op::Dispatch {
            kernel: Rc::clone(kernel),
            buffers: buffers.iter().map(|&b| Rc::clone(b)).collect(),
            push: push.to_vec(),
            groups,
        });
    }

    /// Records a copy of `bytes` bytes from `src` at byte `src_offset` to
    /// `dst` at byte `dst_offset`, ordered like a kernel.
    ///
    /// # Safety
    ///
    /// Both ranges must lie inside their buffers.
    pub(crate) unsafe fn copy(
        &mut self,
        src: &Rc<Allocation>,
        src_offset: usize,
        dst: &Rc<Allocation>,
        dst_offset: usize,
        bytes: usize,
    ) {
        self.ops.push(Op::Copy {
            src: Rc::clone(src),
            src_offset,
            dst: Rc::clone(dst),
            dst_offset,
            bytes,
        });
    }
}

/// Commands recorded once and run many times, like a CUDA graph. Buffers
/// and constants are fixed at recording; what changes between runs is the
/// contents of the buffers. A Metal command buffer runs once, so each run
/// encodes the list again.
pub struct Graph {
    device: Rc<Device>,
    ops: Vec<Op>,
}

impl Graph {
    /// Runs the commands and waits for them; every write is then visible
    /// to the host.
    pub fn launch(&self) -> Result<()> {
        run(&self.device, &self.ops)
    }
}

/// The encoder commands are being written to. Dispatches and copies need
/// different ones, and only one may be open.
enum Encoder {
    None,
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
}

impl Encoder {
    fn end(&mut self) {
        match std::mem::replace(self, Encoder::None) {
            Encoder::None => {}
            Encoder::Compute(e) => e.endEncoding(),
            Encoder::Blit(e) => e.endEncoding(),
        }
    }
}

/// Encodes `ops` into one command buffer, runs it and waits for it.
fn run(device: &Device, ops: &[Op]) -> Result<()> {
    if ops.is_empty() {
        return Ok(());
    }
    let fail = |call, why: &str| Error::Metal {
        call,
        why: why.into(),
    };
    let commands = device
        .queue
        .commandBuffer()
        .ok_or_else(|| fail("commandBuffer", "none"))?;
    let mut encoder = Encoder::None;
    for op in ops {
        match op {
            Op::Dispatch {
                kernel,
                buffers,
                push,
                groups,
            } => {
                if !matches!(encoder, Encoder::Compute(_)) {
                    encoder.end();
                    encoder = Encoder::Compute(
                        commands
                            .computeCommandEncoder()
                            .ok_or_else(|| fail("computeCommandEncoder", "none"))?,
                    );
                }
                let Encoder::Compute(e) = &encoder else {
                    unreachable!()
                };
                e.setComputePipelineState(&kernel.state);
                // SAFETY: the buffers are alive, held by the list; the push
                // constants and sizes are copied by Metal during the call.
                // `Commands::dispatch` vouches for what the shader does.
                unsafe {
                    for (i, b) in buffers.iter().enumerate() {
                        e.setBuffer_offset_atIndex(Some(&b.buffer), 0, i);
                    }
                    if !push.is_empty() {
                        e.setBytes_length_atIndex(bytes_of(push), 4 * push.len(), buffers.len());
                    }
                    e.setBytes_length_atIndex(
                        bytes_of(&ARRAY_SIZES),
                        size_of_val(&ARRAY_SIZES),
                        buffers.len() + 1,
                    );
                    for (i, &bytes) in kernel.threadgroup_bytes.iter().enumerate() {
                        e.setThreadgroupMemoryLength_atIndex(bytes, i);
                    }
                }
                e.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: groups[0] as usize,
                        height: groups[1] as usize,
                        depth: groups[2] as usize,
                    },
                    MTLSize {
                        width: kernel.threads,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            Op::Copy {
                src,
                src_offset,
                dst,
                dst_offset,
                bytes,
            } => {
                if !matches!(encoder, Encoder::Blit(_)) {
                    encoder.end();
                    encoder = Encoder::Blit(
                        commands
                            .blitCommandEncoder()
                            .ok_or_else(|| fail("blitCommandEncoder", "none"))?,
                    );
                }
                let Encoder::Blit(e) = &encoder else {
                    unreachable!()
                };
                // SAFETY: `Commands::copy` vouches for both ranges.
                unsafe {
                    e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &src.buffer,
                        *src_offset,
                        &dst.buffer,
                        *dst_offset,
                        *bytes,
                    )
                };
            }
        }
    }
    encoder.end();
    commands.commit();
    commands.waitUntilCompleted();
    match commands.error() {
        Some(e) => Err(Error::Metal {
            call: "commit",
            why: e.localizedDescription().to_string(),
        }),
        None => Ok(()),
    }
}

fn bytes_of(words: &[u32]) -> NonNull<c_void> {
    NonNull::from(words).cast()
}
