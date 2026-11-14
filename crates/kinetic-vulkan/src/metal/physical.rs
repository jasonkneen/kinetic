use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::info::{DeviceInfo, DeviceKind, Features};

/// The SIMD group size every kernel is written for, the subgroup of the
/// WGSL they come from.
pub(crate) const SUBGROUP: u32 = 32;

/// Apple's PCI vendor id.
const APPLE: u32 = 0x106B;

/// A Metal device before it is opened.
#[derive(Clone)]
pub struct PhysicalDevice {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLDevice>>,
    info: DeviceInfo,
}

impl PhysicalDevice {
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub(crate) fn query(raw: &Retained<ProtocolObject<dyn MTLDevice>>) -> PhysicalDevice {
        // Every Apple GPU runs SIMD groups of 32 with the reductions the
        // kernels use; `Kernel::new` checks the width of each pipeline.
        let features = Features {
            subgroup_size: SUBGROUP,
            subgroup_size_range: (SUBGROUP, SUBGROUP),
            subgroup_arithmetic: true,
            subgroup_size_32: true,
            integer_dot_product: true,
            storage_16bit: true,
            storage_8bit: true,
            push_descriptor: true,
            ..Features::default()
        };
        let info = DeviceInfo {
            name: raw.name().to_string(),
            kind: if raw.hasUnifiedMemory() {
                DeviceKind::Integrated
            } else {
                DeviceKind::Discrete
            },
            vendor_id: APPLE,
            device_id: 0,
            // What `DeviceInfo::missing` asks of a Vulkan driver; Metal has
            // no such number.
            api_version: (1, 3, 0),
            driver_name: "Metal".into(),
            driver_info: String::new(),
            device_local_bytes: raw.recommendedMaxWorkingSetSize(),
            max_buffer_bytes: raw.maxBufferLength() as u64,
            timestamp_period: 0.0,
            features,
        };
        PhysicalDevice {
            raw: raw.clone(),
            info,
        }
    }
}
