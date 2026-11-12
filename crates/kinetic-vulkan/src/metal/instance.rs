use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use crate::error::{Error, Result};
use crate::physical::PhysicalDevice;

// A command-line program gets no default device unless CoreGraphics is
// linked, which nothing else here would do.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// The system's Metal device.
pub struct Instance {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl Instance {
    /// Asks Metal for the default GPU. Fails with [`Error::LoaderNotFound`]
    /// when the system has none.
    pub fn new() -> Result<Rc<Instance>> {
        let raw = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| Error::LoaderNotFound("Metal reports no GPU".into()))?;
        Ok(Rc::new(Instance { raw }))
    }

    /// The default GPU, the only one an Apple Silicon Mac has.
    pub fn physical_devices(self: &Rc<Self>) -> Result<Vec<PhysicalDevice>> {
        Ok(vec![PhysicalDevice::query(&self.raw)])
    }
}
