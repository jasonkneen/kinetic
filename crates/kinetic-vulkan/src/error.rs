use std::fmt;

#[cfg(not(metal_driver))]
use ash::vk;
use kinetic_quant::TensorType;

/// Errors from the Vulkan loader and driver and from this crate's own checks.
#[derive(Clone, Debug)]
pub enum Error {
    /// The Vulkan loader library could not be opened.
    LoaderNotFound(String),
    /// A Vulkan call failed.
    #[cfg(not(metal_driver))]
    Vulkan {
        call: &'static str,
        result: vk::Result,
    },
    /// A Metal call failed.
    #[cfg(metal_driver)]
    Metal { call: &'static str, why: String },
    /// The device lacks something the kernels need; the message says what.
    DeviceUnsupported(String),
    /// No GEMV kernel exists for this weight type.
    Unsupported(TensorType),
    /// Sizes that do not fit together.
    Shape(String),
}

/// `Result` with this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::LoaderNotFound(why) => write!(f, "Vulkan loader not found: {why}"),
            #[cfg(not(metal_driver))]
            Error::Vulkan { call, result } => write!(f, "{call} failed: {result}"),
            #[cfg(metal_driver)]
            Error::Metal { call, why } => write!(f, "{call} failed: {why}"),
            Error::DeviceUnsupported(why) => write!(f, "device not supported: {why}"),
            Error::Unsupported(ty) => write!(f, "no Vulkan GEMV kernel for {ty}"),
            Error::Shape(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Error {}

/// Maps a failed call's result into an [`Error`] naming the call.
#[cfg(not(metal_driver))]
pub(crate) trait Check<T> {
    fn check(self, call: &'static str) -> Result<T>;
}

#[cfg(not(metal_driver))]
impl<T> Check<T> for std::result::Result<T, vk::Result> {
    fn check(self, call: &'static str) -> Result<T> {
        self.map_err(|result| Error::Vulkan { call, result })
    }
}
