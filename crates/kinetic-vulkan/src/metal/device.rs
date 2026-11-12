use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};

use crate::error::{Error, Result};
use crate::info::DeviceInfo;
use crate::physical::PhysicalDevice;

/// An opened device with one command queue.
///
/// Everything made from a device holds an `Rc` to it; the queue is used
/// from one thread at a time, which `Rc` enforces.
pub struct Device {
    physical: PhysicalDevice,
    pub(crate) raw: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl Device {
    /// Opens `physical`, or fails with [`Error::DeviceUnsupported`] when it
    /// lacks what the kernels need (see [`DeviceInfo::missing`]).
    pub fn new(physical: &PhysicalDevice) -> Result<Rc<Device>> {
        if let Some(why) = physical.info().missing() {
            return Err(Error::DeviceUnsupported(why));
        }
        let raw = physical.raw.clone();
        let queue = raw.newCommandQueue().ok_or_else(|| Error::Metal {
            call: "newCommandQueue",
            why: "no queue".into(),
        })?;
        Ok(Rc::new(Device {
            physical: physical.clone(),
            raw,
            queue,
        }))
    }

    pub fn info(&self) -> &DeviceInfo {
        self.physical.info()
    }
}
