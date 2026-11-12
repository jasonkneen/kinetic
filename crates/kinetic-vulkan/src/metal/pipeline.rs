use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLanguageVersion, MTLLibrary,
};

use crate::device::Device;
use crate::error::{Error, Result};
use crate::physical::SUBGROUP;

/// What naga names the entry point `main`, which Metal reserves.
const ENTRY: &str = "main_";

/// A compute pipeline whose buffers are set with each dispatch and whose
/// parameters arrive as one more buffer after them.
pub(crate) struct Kernel {
    pub(crate) state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) bindings: usize,
    pub(crate) push_words: usize,
    /// Threads of one threadgroup, the WGSL's workgroup size.
    pub(crate) threads: usize,
    /// Bytes of each block of threadgroup memory, the WGSL's workgroup
    /// variables, in the order the entry point takes them.
    pub(crate) threadgroup_bytes: Vec<usize>,
}

impl Kernel {
    /// Builds the entry point of `source`, the Metal shader build.rs wrote,
    /// which binds `bindings` buffers and reads `push_words` 32-bit
    /// parameters. Every kernel runs in SIMD groups of [`SUBGROUP`].
    pub(crate) fn new(
        device: &Rc<Device>,
        name: &str,
        source: &[u8],
        bindings: usize,
        push_words: usize,
    ) -> Result<Rc<Kernel>> {
        let fail = |call, why: String| Error::Metal {
            call,
            why: format!("{name}: {why}"),
        };
        let text = std::str::from_utf8(source)
            .map_err(|e| Error::Shape(format!("embedded Metal source of {name}: {e}")))?;
        let threads = threads_per_group(text)
            .ok_or_else(|| Error::Shape(format!("{name} declares no threadgroup size")))?;
        let threadgroup_bytes = threadgroup_bytes(text)
            .ok_or_else(|| Error::Shape(format!("{name} lists no threadgroup memory")))?;
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_0);
        // Fast math lets the compiler reorder sums, and a token must get the
        // same sum alone as in a batch. `mathMode` replaces this setter only
        // from macOS 15 on.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        let library = device
            .raw
            .newLibraryWithSource_options_error(&NSString::from_str(text), Some(&options))
            .map_err(|e| fail("newLibraryWithSource", e.localizedDescription().to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(ENTRY))
            .ok_or_else(|| fail("newFunctionWithName", format!("no {ENTRY}")))?;
        let state = device
            .raw
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                fail(
                    "newComputePipelineState",
                    e.localizedDescription().to_string(),
                )
            })?;
        if state.threadExecutionWidth() != SUBGROUP as usize {
            return Err(Error::DeviceUnsupported(format!(
                "needs SIMD groups of {SUBGROUP}, {name} runs in groups of {}",
                state.threadExecutionWidth()
            )));
        }
        Ok(Rc::new(Kernel {
            state,
            bindings,
            push_words,
            threads,
            threadgroup_bytes,
        }))
    }
}

/// The sizes build.rs listed on the first line.
fn threadgroup_bytes(source: &str) -> Option<Vec<usize>> {
    source
        .lines()
        .next()?
        .strip_prefix("// threadgroup bytes: ")?
        .split_whitespace()
        .map(|n| n.parse().ok())
        .collect()
}

/// The threadgroup size naga wrote on the entry point.
fn threads_per_group(source: &str) -> Option<usize> {
    let attribute = "max_total_threads_per_threadgroup(";
    let rest = &source[source.find(attribute)? + attribute.len()..];
    rest[..rest.find(')')?].parse().ok()
}
