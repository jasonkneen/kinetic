//! What a device is and offers, whichever driver reports it.

/// What kind of processor a device is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

impl DeviceKind {
    /// Sort key: lower is preferred.
    #[cfg_attr(metal_driver, allow(dead_code))]
    pub(crate) fn preference(self) -> u8 {
        match self {
            DeviceKind::Discrete => 0,
            DeviceKind::Integrated => 1,
            DeviceKind::Virtual => 2,
            DeviceKind::Other => 3,
            DeviceKind::Cpu => 4,
        }
    }
}

/// Optional capabilities found on a device. Those the kernels rely on are
/// listed by [`DeviceInfo::missing`] when absent.
#[derive(Clone, Debug, Default)]
pub struct Features {
    /// The subgroup size compute shaders get by default.
    pub subgroup_size: u32,
    /// The smallest and largest subgroup sizes a pipeline can ask for.
    pub subgroup_size_range: (u32, u32),
    /// Subgroup reductions such as `subgroupAdd` in compute shaders.
    pub subgroup_arithmetic: bool,
    /// Compute pipelines run with full subgroups of 32: either they can
    /// require that size, or it is the only size the device has.
    pub subgroup_size_32: bool,
    /// Compute pipelines can require a subgroup size. Without it (the
    /// Vulkan drivers on Metal) the size is the device's own.
    pub subgroup_size_required: bool,
    /// Packed int8 dot products (`VK_KHR_shader_integer_dot_product`).
    pub integer_dot_product: bool,
    /// The driver reports packed signed int8 dot products as hardware-accelerated.
    pub integer_dot_product_accelerated: bool,
    /// 16-bit types in storage buffers.
    pub storage_16bit: bool,
    /// 8-bit types in storage buffers.
    pub storage_8bit: bool,
    /// Cooperative matrix operations (`VK_KHR_cooperative_matrix`).
    pub cooperative_matrix: bool,
    /// Descriptors pushed straight into command buffers (`VK_KHR_push_descriptor`).
    pub push_descriptor: bool,
    /// The compute queue can write timestamps.
    pub timestamps: bool,
}

/// Properties of a physical device.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub kind: DeviceKind,
    pub vendor_id: u32,
    pub device_id: u32,
    /// The Vulkan version the driver supports, as (major, minor, patch).
    pub api_version: (u32, u32, u32),
    pub driver_name: String,
    /// The driver's own version string.
    pub driver_info: String,
    /// Size of the largest device-local memory heap.
    pub device_local_bytes: u64,
    /// The largest storage buffer a kernel can bind.
    pub max_buffer_bytes: u64,
    /// Seconds per timestamp tick.
    pub timestamp_period: f64,
    pub features: Features,
}

impl DeviceInfo {
    /// Why this crate's kernels cannot run on the device, or `None` when they can.
    pub fn missing(&self) -> Option<String> {
        let f = &self.features;
        let (major, minor, _) = self.api_version;
        let mut lacks = Vec::new();
        if (major, minor) < (1, 3) {
            lacks.push(format!("Vulkan 1.3 (driver offers {major}.{minor})"));
        }
        for (have, what) in [
            (f.subgroup_arithmetic, "subgroup arithmetic"),
            (f.subgroup_size_32, "subgroups of 32"),
            (f.integer_dot_product, "integer dot products"),
            (f.push_descriptor, "push descriptors"),
        ] {
            if !have {
                lacks.push(what.to_string());
            }
        }
        (!lacks.is_empty()).then(|| format!("needs {}", lacks.join(", ")))
    }
}
