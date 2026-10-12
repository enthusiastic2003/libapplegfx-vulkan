//! libapplegfx-vulkan's C interface over metal2vulkan's translator (see
//! `src/m2v/lagfx_m2v.h` for the contract). Every entry point catches panics
//! and reports errors as text; handles are opaque boxes the caller frees.

use std::collections::BTreeSet;
use std::ffi::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

use metal2vulkan::device::{ApiVersion, Device, DeviceDescription, DevicePolicy, EnableSet};
use metal2vulkan::library::Library;
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, VertexFormat, VertexStepFunction,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipeline, RenderPipelineDescriptor,
    StageSource, VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan::Budget;

pub struct M2vDevice {
    device: Device,
}

pub struct M2vPipeline {
    pipeline: RenderPipeline,
    /// The descriptor it was compiled from, for its variants.
    desc: RenderPipelineDescriptor,
    /// Variants compiled for draws whose plan asked for one, by key.
    variants: std::sync::Mutex<Vec<(metal2vulkan::cache::VariantKey, Box<M2vPipeline>)>>,
}

fn set_err(err: *mut c_char, len: usize, msg: &str) {
    if err.is_null() || len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(len - 1);
    // SAFETY: the caller gives `len` writable bytes at `err`.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), err as *mut u8, n);
        *err.add(n) = 0;
    }
}

fn guard<T>(err: *mut c_char, len: usize, f: impl FnOnce() -> Result<T, String>) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Some(v),
        Ok(Err(e)) => {
            set_err(err, len, &e);
            None
        }
        Err(_) => {
            set_err(err, len, "metal2vulkan panicked");
            None
        }
    }
}

/// Describe the physical device for the translator. The caller's `VkDevice`
/// must enable every core (1.0-1.3) feature the device supports; no extension
/// is assumed, so the translator uses core features only.
///
/// # Safety
/// `get_instance_proc_addr` is the loader's `vkGetInstanceProcAddr`, `instance` a live
/// `VkInstance` created with at least `api_version`, `physical_device` one of its devices.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_device_new(
    get_instance_proc_addr: *const c_void,
    instance: u64,
    physical_device: u64,
    api_version: u32,
    err: *mut c_char,
    err_len: usize,
) -> *mut M2vDevice {
    let r = guard(err, err_len, || {
        if get_instance_proc_addr.is_null() {
            return Err("no vkGetInstanceProcAddr".into());
        }
        let static_fn = ash::StaticFn {
            // SAFETY: the caller passes the loader's vkGetInstanceProcAddr.
            get_instance_proc_addr: unsafe { std::mem::transmute(get_instance_proc_addr) },
        };
        use ash::vk::Handle;
        // SAFETY: a live instance (caller's contract).
        let inst = unsafe { ash::Instance::load(&static_fn, ash::vk::Instance::from_raw(instance)) };
        let api = ApiVersion::from_packed(api_version);
        // SAFETY: caller's contract.
        let desc = unsafe {
            DeviceDescription::query(&inst, ash::vk::PhysicalDevice::from_raw(physical_device), api)
        }
        .map_err(|e| format!("device query: {e}"))?;
        let policy = DevicePolicy::default();
        let recommended = desc.recommended_enables(&policy);
        let api = desc.api_version;
        let none = BTreeSet::new();
        let mut enabled = EnableSet::new();
        enabled.features = recommended
            .features
            .iter()
            .copied()
            .filter(|f| f.exposure().exposed(api, &none))
            .collect();
        let desc = desc
            .with_enabled(&enabled)
            .map_err(|e| format!("enables: {e}"))?;
        let device = Device::new(desc, policy).map_err(|e| format!("device: {e}"))?;
        Ok(Box::into_raw(Box::new(M2vDevice { device })))
    });
    r.unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_device_free(dev: *mut M2vDevice) {
    if !dev.is_null() {
        // SAFETY: a handle from lagfx_m2v_device_new.
        drop(unsafe { Box::from_raw(dev) });
    }
}

/// One vertex attribute / layout of a render pipeline's vertex descriptor, raw Metal values.
#[repr(C)]
pub struct LagfxM2vVertexAttribute {
    pub index: u32,
    pub format: u32, // MTLVertexFormat
    pub offset: u32,
    pub buffer_index: u32,
}

#[repr(C)]
pub struct LagfxM2vVertexLayout {
    pub buffer_index: u32,
    pub stride: u32,        // 0 = MTLBufferLayoutStrideDynamic
    pub step_function: u32, // MTLVertexStepFunction
    pub step_rate: u32,
}

/// A render pipeline: the stages' libraries (one function each, as the guest's
/// pipeline objects reference them) and colour attachment 0, raw Metal values.
#[repr(C)]
pub struct LagfxM2vRenderDesc {
    pub vertex_lib: *const u8,
    pub vertex_lib_len: usize,
    pub fragment_lib: *const u8, // NULL: no fragment stage
    pub fragment_lib_len: usize,
    pub color_format: u32, // MTLPixelFormat; 0 = no colour attachment
    pub blending: u32,
    pub src_rgb: u32,
    pub dst_rgb: u32,
    pub rgb_op: u32,
    pub src_alpha: u32,
    pub dst_alpha: u32,
    pub alpha_op: u32,
    pub write_mask: u32,
    pub has_vertex_descriptor: u32,
    pub n_attributes: u32,
    pub attributes: *const LagfxM2vVertexAttribute,
    pub n_layouts: u32,
    pub layouts: *const LagfxM2vVertexLayout,
}

fn only_function(bytes: &[u8]) -> Result<metal2vulkan::library::Function, String> {
    let lib = Library::parse(bytes).map_err(|e| format!("library: {e}"))?;
    let name = lib
        .functions()
        .next()
        .ok_or("library has no function")?
        .name
        .to_string();
    lib.function(&name).map_err(|e| format!("function {name}: {e}"))
}

fn raw<T: TryFrom<u32>>(v: u32, what: &str) -> Result<T, String> {
    T::try_from(v).map_err(|_| format!("unknown {what} {v}"))
}

/// Compile a render pipeline. NULL on failure (or refusal), with the reason in `err`.
///
/// # Safety
/// `dev` from lagfx_m2v_device_new; `desc` and every array/buffer it points at valid.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_compile_render(
    dev: *const M2vDevice,
    desc: *const LagfxM2vRenderDesc,
    err: *mut c_char,
    err_len: usize,
) -> *mut M2vPipeline {
    let r = guard(err, err_len, || {
        // SAFETY: caller's contract.
        let (dev, d) = unsafe { (dev.as_ref().ok_or("no device")?, desc.as_ref().ok_or("no desc")?) };
        if d.vertex_lib.is_null() {
            return Err("no vertex library".into());
        }
        // SAFETY: caller's contract.
        let vlib = unsafe { std::slice::from_raw_parts(d.vertex_lib, d.vertex_lib_len) };
        let mut rp = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
            only_function(vlib)?,
        )));
        if !d.fragment_lib.is_null() {
            // SAFETY: caller's contract.
            let flib = unsafe { std::slice::from_raw_parts(d.fragment_lib, d.fragment_lib_len) };
            rp.fragment = Some(StageSource::Function(FunctionDescriptor::new(only_function(flib)?)));
        }
        if d.color_format != 0 {
            let mut c = ColorAttachmentDescriptor::new(raw::<PixelFormat>(d.color_format, "pixel format")?);
            c.blending_enabled = d.blending != 0;
            c.source_rgb_blend_factor = raw::<BlendFactor>(d.src_rgb, "blend factor")?;
            c.destination_rgb_blend_factor = raw::<BlendFactor>(d.dst_rgb, "blend factor")?;
            c.rgb_blend_operation = raw::<BlendOperation>(d.rgb_op, "blend operation")?;
            c.source_alpha_blend_factor = raw::<BlendFactor>(d.src_alpha, "blend factor")?;
            c.destination_alpha_blend_factor = raw::<BlendFactor>(d.dst_alpha, "blend factor")?;
            c.alpha_blend_operation = raw::<BlendOperation>(d.alpha_op, "blend operation")?;
            c.write_mask = raw::<ColorWriteMask>(d.write_mask, "write mask")?;
            rp.color_attachments[0] = Some(c);
        }
        if d.has_vertex_descriptor != 0 {
            let mut vd = VertexDescriptor::new();
            if d.n_attributes > 0 {
                // SAFETY: caller's contract.
                let attrs = unsafe { std::slice::from_raw_parts(d.attributes, d.n_attributes as usize) };
                for a in attrs {
                    vd.attributes.insert(
                        a.index,
                        VertexAttributeDescriptor::new(
                            raw::<VertexFormat>(a.format, "vertex format")?,
                            a.offset,
                            a.buffer_index,
                        ),
                    );
                }
            }
            if d.n_layouts > 0 {
                // SAFETY: caller's contract.
                let lays = unsafe { std::slice::from_raw_parts(d.layouts, d.n_layouts as usize) };
                for l in lays {
                    let mut ld = VertexBufferLayoutDescriptor::new(if l.stride == 0 { None } else { Some(l.stride) });
                    ld.step_function = raw::<VertexStepFunction>(l.step_function, "step function")?;
                    ld.step_rate = l.step_rate;
                    vd.layouts.insert(l.buffer_index, ld);
                }
            }
            rp.vertex_descriptor = Some(vd);
        }
        let pipeline = dev
            .device
            .compile_render_pipeline(&rp, &Budget::standard())
            .map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(M2vPipeline {
            pipeline,
            desc: rp,
            variants: std::sync::Mutex::new(Vec::new()),
        })))
    });
    r.unwrap_or(std::ptr::null_mut())
}

/// Number of shader stages (vertex, then fragment when rasterizing).
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_pipeline_stage_count(p: *const M2vPipeline) -> u32 {
    // SAFETY: a handle from lagfx_m2v_compile_render.
    unsafe { p.as_ref() }.map_or(0, |p| p.pipeline.stages.len() as u32)
}

/// Stage `i`'s SPIR-V (words), `*n_words` set; NULL past the last stage.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_pipeline_spirv(
    p: *const M2vPipeline,
    i: u32,
    n_words: *mut usize,
) -> *const u32 {
    // SAFETY: a handle from lagfx_m2v_compile_render; `n_words` writable.
    let Some(p) = (unsafe { p.as_ref() }) else { return std::ptr::null() };
    let Some(s) = p.pipeline.stages.get(i as usize) else { return std::ptr::null() };
    if !n_words.is_null() {
        unsafe { *n_words = s.spirv.len() };
    }
    s.spirv.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_pipeline_free(p: *mut M2vPipeline) {
    if !p.is_null() {
        // SAFETY: a handle from lagfx_m2v_compile_render.
        drop(unsafe { Box::from_raw(p) });
    }
}

// ===================================================================== pipeline facts

use ash::vk as avk;
use metal2vulkan::bind::{
    sampler_desc, BindState, BoundBuffer, BoundSampler, BoundTexture, DrawCall, DrawCommands,
    ImageSource, SamplerDescriptor, SamplerSource, StageTable, TextureFacts, WriteSource,
};
use metal2vulkan::bind::NeutralResource;
use metal2vulkan::metal::{
    CompareFunction, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, SamplerMipFilter,
    TextureType, TextureUsage,
};
use metal2vulkan::pipeline::Stage;

macro_rules! raw {
    ($t:ident, $v:expr) => {
        avk::$t::from($v).as_raw() as u32
    };
}

/// One binding of the pipeline layout.
#[repr(C)]
pub struct LagfxM2vLayoutBinding {
    pub set: u32,
    pub binding: u32,
    pub descriptor_type: u32, // VkDescriptorType
    pub count: u32,
    pub stages: u32,          // VkShaderStageFlags
    pub flags: u32,           // VkDescriptorBindingFlags
    pub has_immutable_sampler: u32,
    pub immutable_sampler: LagfxM2vVkSampler,
}

/// A Vulkan sampler description, raw Vulkan values.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct LagfxM2vVkSampler {
    pub mag_filter: u32,
    pub min_filter: u32,
    pub mipmap_mode: u32,
    pub address_u: u32,
    pub address_v: u32,
    pub address_w: u32,
    pub mip_lod_bias: f32,
    pub anisotropy_enable: u32,
    pub max_anisotropy: f32,
    pub compare_enable: u32,
    pub compare_op: u32,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: u32,
    pub reduction: u32,
}

fn vk_sampler(d: &metal2vulkan::vk::SamplerDesc) -> LagfxM2vVkSampler {
    LagfxM2vVkSampler {
        mag_filter: raw!(Filter, d.mag_filter),
        min_filter: raw!(Filter, d.min_filter),
        mipmap_mode: raw!(SamplerMipmapMode, d.mipmap_mode),
        address_u: raw!(SamplerAddressMode, d.address_mode_u),
        address_v: raw!(SamplerAddressMode, d.address_mode_v),
        address_w: raw!(SamplerAddressMode, d.address_mode_w),
        mip_lod_bias: d.mip_lod_bias,
        anisotropy_enable: d.max_anisotropy.is_some() as u32,
        max_anisotropy: d.max_anisotropy.unwrap_or(1.0),
        compare_enable: d.compare.is_some() as u32,
        compare_op: d.compare.map_or(0, |c| raw!(CompareOp, c)),
        min_lod: d.min_lod,
        max_lod: d.max_lod,
        border_color: raw!(BorderColor, d.border_color),
        reduction: raw!(SamplerReductionMode, d.reduction),
    }
}

/// The pipeline's layout bindings; returns the count, fills up to `cap`.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_pipeline_layout(
    p: *const M2vPipeline,
    out: *mut LagfxM2vLayoutBinding,
    cap: u32,
    push_size: *mut u32,
    push_stages: *mut u32,
) -> u32 {
    // SAFETY: a handle from lagfx_m2v_compile_render; `out` holds `cap` entries.
    let Some(p) = (unsafe { p.as_ref() }) else { return 0 };
    let l = &p.pipeline.layout;
    let mut n = 0u32;
    for set in &l.sets {
        for b in &set.bindings {
            if n < cap && !out.is_null() {
                let e = LagfxM2vLayoutBinding {
                    set: set.set,
                    binding: b.binding,
                    descriptor_type: raw!(DescriptorType, b.ty),
                    count: b.count,
                    stages: raw!(ShaderStageFlags, b.stages),
                    flags: raw!(DescriptorBindingFlags, b.flags),
                    has_immutable_sampler: b.immutable_sampler.is_some() as u32,
                    immutable_sampler: b.immutable_sampler.as_ref().map(vk_sampler).unwrap_or_default(),
                };
                unsafe { out.add(n as usize).write(e) };
            }
            n += 1;
        }
    }
    let (mut size, mut stages) = (0u32, 0u32);
    for r in &l.push_constants {
        size = size.max(r.offset + r.size);
        stages |= raw!(ShaderStageFlags, r.stages);
    }
    if !push_size.is_null() { unsafe { *push_size = size } }
    if !push_stages.is_null() { unsafe { *push_stages = stages } }
    n
}

/// Stage `i`'s entry point name (NUL-terminated into `out`).
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_pipeline_entry(
    p: *const M2vPipeline, i: u32, out: *mut c_char, cap: usize,
) -> u32 {
    let Some(p) = (unsafe { p.as_ref() }) else { return 0 };
    let Some(s) = p.pipeline.stages.get(i as usize) else { return 0 };
    set_err(out, cap, &s.entry_point);
    raw!(ShaderStageFlags, s.stage)
}

/// Stage `i`'s constant blob (its `ConstantBlob` binding's data), NULL when none.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_pipeline_constant_blob(
    p: *const M2vPipeline, i: u32, len: *mut usize,
) -> *const u8 {
    let Some(p) = (unsafe { p.as_ref() }) else { return std::ptr::null() };
    let Some(b) = p.pipeline.stages.get(i as usize).and_then(|s| s.constant_blob.as_ref()) else {
        return std::ptr::null();
    };
    if !len.is_null() { unsafe { *len = b.len() } }
    b.as_ptr()
}

#[repr(C)]
pub struct LagfxM2vVertexBinding {
    pub binding: u32,
    pub buffer_index: u32,
    pub input_rate: u32, // VkVertexInputRate
    pub stride: u32,     // 0 = per bind (dynamic)
}

#[repr(C)]
pub struct LagfxM2vVertexAttributeOut {
    pub location: u32,
    pub binding: u32,
    pub format: u32, // VkFormat
    pub offset: u32,
}

/// Vertex input: bindings and attributes (fixed-function fetch). Returns
/// n_bindings; *n_attributes set. Arrays hold up to 16 each.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_pipeline_vertex_input(
    p: *const M2vPipeline,
    bindings: *mut LagfxM2vVertexBinding,
    attributes: *mut LagfxM2vVertexAttributeOut,
    n_attributes: *mut u32,
) -> u32 {
    let Some(p) = (unsafe { p.as_ref() }) else { return 0 };
    let vi = &p.pipeline.vertex_input;
    let mut nb = 0u32;
    for b in vi.bindings.iter().take(16) {
        let stride = match b.stride {
            metal2vulkan::pipeline::VertexStride::Static(s) => s,
            _ => 0,
        };
        unsafe {
            bindings.add(nb as usize).write(LagfxM2vVertexBinding {
                binding: b.binding,
                buffer_index: b.buffer_index,
                input_rate: raw!(VertexInputRate, b.input_rate),
                stride,
            })
        };
        nb += 1;
    }
    let mut na = 0u32;
    for a in vi.attributes.iter().take(16) {
        unsafe {
            attributes.add(na as usize).write(LagfxM2vVertexAttributeOut {
                location: a.location,
                binding: a.binding,
                format: raw!(Format, a.format),
                offset: a.offset,
            })
        };
        na += 1;
    }
    if !n_attributes.is_null() { unsafe { *n_attributes = na } }
    nb
}

/// Primitive topology class the pipeline was compiled for (VkPrimitiveTopology).
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_pipeline_topology(p: *const M2vPipeline) -> u32 {
    unsafe { p.as_ref() }.map_or(3, |p| raw!(PrimitiveTopology, p.pipeline.graphics.topology))
}

// ===================================================================== draws

#[repr(C)]
pub struct LagfxM2vBuffer {
    pub index: u32,
    pub is_bytes: u32,  // set*Bytes: `length` bytes
    pub length: u64,
    pub offset: u64,
    pub gpu_address: u64,
    pub device_address: u64, // host VkBuffer device address of the allocation; 0 = none
}

#[repr(C)]
pub struct LagfxM2vTexture {
    pub index: u32,
    pub texture_type: u32, // MTLTextureType
    pub pixel_format: u32, // MTLPixelFormat
    pub usage: u32,        // MTLTextureUsage
    pub mip_levels: u32,
    pub array_length: u32,
    pub sample_count: u32,
}

#[repr(C)]
pub struct LagfxM2vSampler {
    pub index: u32,
    pub min_filter: u32, // MTLSamplerMinMagFilter
    pub mag_filter: u32,
    pub mip_filter: u32, // MTLSamplerMipFilter
    pub s_address: u32,  // MTLSamplerAddressMode
    pub t_address: u32,
    pub r_address: u32,
    pub border_color: u32, // MTLSamplerBorderColor
    pub normalized: u32,
    pub max_anisotropy: u32,
    pub compare_function: u32, // MTLCompareFunction (0 = never: no comparison)
    pub lod_min: f32,
    pub lod_max: f32,
}

#[repr(C)]
pub struct LagfxM2vStage {
    pub n_buffers: u32,
    pub buffers: *const LagfxM2vBuffer,
    pub n_textures: u32,
    pub textures: *const LagfxM2vTexture,
    pub n_samplers: u32,
    pub samplers: *const LagfxM2vSampler,
}

pub const W_UNSUPPORTED: u32 = 0;
pub const W_BUFFER: u32 = 1;
pub const W_INLINE_BYTES: u32 = 2;
pub const W_TEXTURE: u32 = 3;
pub const W_SAMPLER: u32 = 4;
pub const W_COMBINED: u32 = 5;
pub const W_NEUTRAL: u32 = 6;
pub const W_CONSTANT_BLOB: u32 = 7;
pub const W_BYTES: u32 = 8;

/// One descriptor write of a draw.
#[repr(C)]
#[derive(Default)]
pub struct LagfxM2vWrite {
    pub set: u32,
    pub binding: u32,
    pub element: u32,
    pub descriptor_type: u32,
    pub kind: u32,           // W_*
    pub stage: u32,          // slot sources: 0 vertex, 1 fragment
    pub slot: u32,
    // image view (W_TEXTURE, W_COMBINED with a texture)
    pub view_type: u32,
    pub view_format: u32,
    pub view_layout: u32,
    pub view_aspect: u32,
    pub swizzle: [u32; 4],
    // sampler (W_SAMPLER, W_COMBINED): 0 = this description, 1 = immutable, 2 = neutral
    pub sampler_kind: u32,
    pub sampler: LagfxM2vVkSampler,
    // neutral (W_NEUTRAL, or a combined write's neutral image when image_neutral)
    pub image_neutral: u32,
    pub neutral_kind: u32,       // 1 zero buffer, 2 sink, 3 image, 4 texel buffer, 5 sampler
    pub neutral_min_bytes: u32,
    pub neutral_view_type: u32,
    pub neutral_sampled_type: u32, // 0 float, 1 sint, 2 uint, 3 uint64
    pub neutral_format: u32,       // storage image / texel buffer format, 0 = sampled image
    pub neutral_compare: u32,
    // buffer range (W_BUFFER, W_INLINE_BYTES, W_BYTES)
    pub has_range: u32,
    pub range_offset: u64,
    pub range_size: u64,
    // W_BYTES / W_CONSTANT_BLOB: index into lagfx_m2v_draw_bytes
    pub bytes_index: i32,
}

#[repr(C)]
pub struct LagfxM2vPush {
    pub stages: u32,
    pub offset: u32,
    pub size: u32,
    pub data_offset: u32, // into lagfx_m2v_draw_push_data
}

pub struct M2vDraw {
    /// The pipeline the draw was planned on: the one given, or its variant.
    pipeline: *const M2vPipeline,
    writes: Vec<LagfxM2vWrite>,
    blobs: Vec<Vec<u8>>,
    push: Vec<LagfxM2vPush>,
    push_data: Vec<u8>,
    spec_data: Vec<u8>,
    spec_entries: Vec<[u32; 3]>, // constant id, offset, size
    vertex_buffers: Vec<[u64; 3]>, // binding, Metal buffer index, offset
    /// [kind 0 draw / 1 indexed, count, instance_count, first (vertex or index),
    ///  vertex offset (indexed), first_instance, index type (VkIndexType), unused]
    draw: [u32; 8],
    index_offset: u64,            // indexed: the index buffer's bind offset
    holes: Vec<[u32; 3]>,          // kind (1 root table, 2 execution status), stages, offset
    root_tables: Vec<(u32, Vec<u8>)>, // stages, bytes
}

fn stage_of(s: Stage) -> u32 {
    match s {
        Stage::Vertex => 0,
        Stage::Fragment => 1,
        _ => 9,
    }
}

fn neutral(w: &mut LagfxM2vWrite, n: &NeutralResource) {
    match *n {
        NeutralResource::ZeroBuffer { min_bytes } => { w.neutral_kind = 1; w.neutral_min_bytes = min_bytes; }
        NeutralResource::Sink { min_bytes } => { w.neutral_kind = 2; w.neutral_min_bytes = min_bytes; }
        NeutralResource::Image { view_type, sampled_type, depth_compare, storage_format, .. } => {
            w.neutral_kind = 3;
            w.neutral_view_type = raw!(ImageViewType, view_type);
            w.neutral_sampled_type = match sampled_type {
                metal2vulkan::pipeline::SampledType::Float => 0,
                metal2vulkan::pipeline::SampledType::Sint => 1,
                metal2vulkan::pipeline::SampledType::Uint => 2,
                _ => 3,
            };
            w.neutral_compare = depth_compare as u32;
            w.neutral_format = storage_format.map_or(0, |f| raw!(Format, f));
        }
        NeutralResource::TexelBuffer { format } => { w.neutral_kind = 4; w.neutral_format = raw!(Format, format); }
        NeutralResource::Sampler { compare } => { w.neutral_kind = 5; w.neutral_compare = compare as u32; }
        _ => {}
    }
}

fn build_state(stages: &[LagfxM2vStage; 2]) -> Result<(BindState, Vec<BTreeMapSampler>), String> {
    let mut state = BindState::default();
    let mut samplers = Vec::new();
    for (si, st) in stages.iter().enumerate() {
        let stage = if si == 0 { Stage::Vertex } else { Stage::Fragment };
        let mut t = StageTable::default();
        // SAFETY: caller's arrays.
        let bufs = if st.n_buffers > 0 { unsafe { std::slice::from_raw_parts(st.buffers, st.n_buffers as usize) } } else { &[] };
        for b in bufs {
            let bound = if b.is_bytes != 0 {
                BoundBuffer::Bytes { len: b.length as u32, attribute_stride: None }
            } else {
                BoundBuffer::Buffer {
                    length: b.length,
                    offset: b.offset,
                    gpu_address: b.gpu_address,
                    device_address: if b.device_address != 0 { Some(b.device_address) } else { None },
                    attribute_stride: None,
                }
            };
            t.buffers.insert(b.index, bound);
        }
        let texs = if st.n_textures > 0 { unsafe { std::slice::from_raw_parts(st.textures, st.n_textures as usize) } } else { &[] };
        for x in texs {
            let mut facts = TextureFacts::new(
                raw::<TextureType>(x.texture_type, "texture type")?,
                raw::<PixelFormat>(x.pixel_format, "pixel format")?,
            );
            facts.usage = TextureUsage::try_from(x.usage).unwrap_or(facts.usage);
            if x.mip_levels > 0 { facts.mipmap_level_count = x.mip_levels; }
            if x.array_length > 0 { facts.array_length = x.array_length; }
            if x.sample_count > 0 { facts.sample_count = x.sample_count; }
            t.textures.insert(x.index, BoundTexture::Texture { facts, resource_id: None });
        }
        let sams = if st.n_samplers > 0 { unsafe { std::slice::from_raw_parts(st.samplers, st.n_samplers as usize) } } else { &[] };
        for s in sams {
            let mut d = SamplerDescriptor::new();
            d.min_filter = raw::<SamplerMinMagFilter>(s.min_filter, "min filter")?;
            d.mag_filter = raw::<SamplerMinMagFilter>(s.mag_filter, "mag filter")?;
            d.mip_filter = raw::<SamplerMipFilter>(s.mip_filter, "mip filter")?;
            d.s_address_mode = raw::<SamplerAddressMode>(s.s_address, "address mode")?;
            d.t_address_mode = raw::<SamplerAddressMode>(s.t_address, "address mode")?;
            d.r_address_mode = raw::<SamplerAddressMode>(s.r_address, "address mode")?;
            d.border_color = raw::<SamplerBorderColor>(s.border_color, "border colour")?;
            d.normalized_coordinates = s.normalized != 0;
            d.lod_min_clamp = s.lod_min;
            d.lod_max_clamp = s.lod_max;
            d.max_anisotropy = s.max_anisotropy.max(1);
            if s.compare_function != 0 {
                d.compare_function = raw::<CompareFunction>(s.compare_function, "compare function")?;
            }
            samplers.push(BTreeMapSampler { stage: si as u32, index: s.index, desc: d });
            t.samplers.insert(s.index, BoundSampler::new(d));
        }
        state.stages.insert(stage, t);
    }
    Ok((state, samplers))
}

struct BTreeMapSampler {
    stage: u32,
    index: u32,
    desc: SamplerDescriptor,
}

fn view_into(w: &mut LagfxM2vWrite, v: &metal2vulkan::bind::ImageViewDesc) {
    w.view_type = raw!(ImageViewType, v.view_type);
    w.view_format = raw!(Format, v.format);
    w.view_layout = raw!(ImageLayout, v.layout);
    w.view_aspect = raw!(ImageAspectFlags, v.aspect);
    let c = avk::ComponentMapping::from(v.components);
    w.swizzle = [c.r.as_raw() as u32, c.g.as_raw() as u32, c.b.as_raw() as u32, c.a.as_raw() as u32];
}

/// Plan and instantiate one non-indexed draw (`drawPrimitives`). NULL on
/// failure or when the draw needs something this bridge doesn't do yet (a
/// pipeline variant, argument buffers): the caller falls back.
/// An indexed draw's index buffer: its type (MTLIndexType: 0 UInt16, 1 UInt32),
/// allocation length, the draw's index offset in bytes, and base vertex.
#[repr(C)]
pub struct LagfxM2vIndexed {
    pub index_type: u32,
    pub buffer_length: u64,
    pub offset: u64,
    pub base_vertex: i32,
}

#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_plan_draw(
    dev: *const M2vDevice,
    p: *const M2vPipeline,
    stages: *const LagfxM2vStage, // [2]: vertex, fragment
    vertex_start: u32,
    vertex_count: u32,
    instance_count: u32,
    base_instance: u32,
    indexed: *const LagfxM2vIndexed, // NULL: drawPrimitives
    err: *mut c_char,
    err_len: usize,
) -> *mut M2vDraw {
    let r = guard(err, err_len, || {
        let (dev, p) = unsafe { (dev.as_ref().ok_or("no device")?, p.as_ref().ok_or("no pipeline")?) };
        let st = unsafe { &*(stages as *const [LagfxM2vStage; 2]) };
        let (state, samplers) = build_state(st)?;
        let shape = state.shape();
        let mut plan = p.pipeline.plan(&dev.device, &shape).map_err(|e| format!("plan: {e}"))?;
        let mut used: *const M2vPipeline = p;
        if !plan.variant.is_empty() {
            let key = plan.variant.clone();
            let mut vars = p.variants.lock().map_err(|_| "variant lock poisoned")?;
            let idx = match vars.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    let v = dev
                        .device
                        .compile_render_variant(&p.desc, &key, &Budget::standard())
                        .map_err(|e| format!("variant: {e}"))?;
                    vars.push((key.clone(), Box::new(M2vPipeline {
                        pipeline: v,
                        desc: p.desc.clone(),
                        variants: std::sync::Mutex::new(Vec::new()),
                    })));
                    vars.len() - 1
                }
            };
            let vp: &M2vPipeline = &vars[idx].1;
            used = vp;
            plan = vp.pipeline.plan(&dev.device, &shape).map_err(|e| format!("variant plan: {e}"))?;
            if plan.variant != key {
                return Err("variant plan asks for another variant".into());
            }
        }
        let draw = match unsafe { indexed.as_ref() } {
            None => DrawCall::Primitives { vertex_start, vertex_count, instance_count, base_instance },
            Some(ix) => DrawCall::Indexed {
                index_type: metal2vulkan::vk::IndexType::from(if ix.index_type == 1 {
                    avk::IndexType::UINT32
                } else {
                    avk::IndexType::UINT16
                }),
                index_buffer: metal2vulkan::bind::IndexBufferRef::new(0, ix.buffer_length),
                index_count: vertex_count,
                index_offset: ix.offset,
                instance_count,
                base_vertex: ix.base_vertex,
                base_instance,
            },
        };
        let cmds: DrawCommands = plan.instantiate_draw(&state, &draw).map_err(|e| format!("draw: {e}"))?;
        let mut holes = Vec::new();
        for h in &cmds.push_holes {
            match *h {
                metal2vulkan::bind::PushHole::RootTableAddress { stages, offset } => {
                    holes.push([1, raw!(ShaderStageFlags, stages), offset])
                }
                metal2vulkan::bind::PushHole::ExecutionStatusAddress { stages, offset } => {
                    holes.push([2, raw!(ShaderStageFlags, stages), offset])
                }
                _ => return Err(format!("unsupported push hole {h:?}")),
            }
        }
        let root_tables: Vec<(u32, Vec<u8>)> = cmds
            .root_tables
            .iter()
            .map(|(st, t)| (raw!(ShaderStageFlags, *st), t.bytes.clone()))
            .collect();
        let mut out = M2vDraw {
            pipeline: used,
            writes: Vec::new(), blobs: Vec::new(), push: Vec::new(), push_data: Vec::new(),
            spec_data: plan.specialization.data.clone(),
            spec_entries: plan.specialization.entries.iter().map(|e| {
                let a = avk::SpecializationMapEntry::from(*e);
                [a.constant_id, a.offset, a.size as u32]
            }).collect(),
            vertex_buffers: Vec::new(), draw: [0; 8], index_offset: 0,
            holes, root_tables,
        };
        let find_sampler = |stage: Stage, index: u32| {
            samplers.iter().find(|s| s.stage == stage_of(stage) && s.index == index).map(|s| s.desc)
        };
        for d in &plan.descriptors {
            let mut w = LagfxM2vWrite {
                set: d.set, binding: d.binding, element: d.array_element,
                descriptor_type: raw!(DescriptorType, d.ty), bytes_index: -1, ..Default::default()
            };
            match &d.source {
                WriteSource::Buffer { slot } => { w.kind = W_BUFFER; w.stage = stage_of(slot.stage); w.slot = slot.index; }
                WriteSource::InlineBytes { slot } => { w.kind = W_INLINE_BYTES; w.stage = stage_of(slot.stage); w.slot = slot.index; }
                WriteSource::Texture { slot, view } => {
                    w.kind = W_TEXTURE; w.stage = stage_of(slot.stage); w.slot = slot.index; view_into(&mut w, view);
                }
                WriteSource::Sampler { slot, compare, integer, opaque_black } => {
                    w.kind = W_SAMPLER; w.stage = stage_of(slot.stage); w.slot = slot.index;
                    let desc = find_sampler(slot.stage, slot.index).ok_or("sampler slot not bound")?;
                    let vs = sampler_desc(&dev.device, &desc, *compare, *integer, *opaque_black, None)
                        .map_err(|e| format!("sampler: {e}"))?;
                    w.sampler = vk_sampler(&vs);
                }
                WriteSource::CombinedImageSampler { image, sampler } => {
                    w.kind = W_COMBINED;
                    let mut fmt = None;
                    match image {
                        ImageSource::Texture { slot, view } => {
                            w.stage = stage_of(slot.stage); w.slot = slot.index; view_into(&mut w, view);
                            fmt = Some(view.format);
                        }
                        ImageSource::Neutral(n) => { w.image_neutral = 1; neutral(&mut w, n); }
                    }
                    match sampler {
                        SamplerSource::Slot { slot, compare, integer, opaque_black } => {
                            let desc = find_sampler(slot.stage, slot.index).ok_or("sampler slot not bound")?;
                            let vs = sampler_desc(&dev.device, &desc, *compare, *integer, *opaque_black, fmt)
                                .map_err(|e| format!("sampler: {e}"))?;
                            w.sampler = vk_sampler(&vs);
                        }
                        SamplerSource::Immutable => w.sampler_kind = 1,
                        SamplerSource::Neutral(_) => w.sampler_kind = 2,
                    }
                }
                WriteSource::Neutral(n) => { w.kind = W_NEUTRAL; neutral(&mut w, n); }
                WriteSource::ConstantBlob { stage } => {
                    w.kind = W_CONSTANT_BLOB;
                    w.stage = stage_of(*stage);
                }
                WriteSource::ArgumentTable { stage, .. } | WriteSource::HandleTable { stage, .. } => {
                    w.kind = W_BYTES;
                    w.stage = stage_of(*stage);
                    if let Some(b) = cmds.descriptor_bytes.iter().find(|b| b.set == d.set && b.binding == d.binding) {
                        w.bytes_index = out.blobs.len() as i32;
                        out.blobs.push(b.bytes.clone());
                    }
                }
                _ => return Err(format!("unsupported descriptor source at set {} binding {}", d.set, d.binding)),
            }
            if let Some(r) = cmds.descriptor_ranges.iter().find(|r| r.set == d.set && r.binding == d.binding) {
                w.has_range = 1;
                w.range_offset = r.range.offset;
                w.range_size = r.range.range;
            }
            out.writes.push(w);
        }
        for pw in &cmds.push_constants {
            out.push.push(LagfxM2vPush {
                stages: raw!(ShaderStageFlags, pw.stages),
                offset: pw.offset,
                size: pw.bytes.len() as u32,
                data_offset: out.push_data.len() as u32,
            });
            out.push_data.extend_from_slice(&pw.bytes);
        }
        for vb in &plan.vertex_buffers {
            let off = cmds.buffer_offsets.iter().find(|(b, _)| *b == vb.binding).map_or(0, |(_, o)| *o);
            out.vertex_buffers.push([vb.binding as u64, vb.buffer_index as u64, off]);
        }
        if cmds.index_overrun.is_some() {
            return Err(format!("index overrun {:?}", cmds.index_overrun));
        }
        match cmds.draw {
            metal2vulkan::vk::DrawCommand::Draw { vertex_count, instance_count, first_vertex, first_instance } => {
                out.draw = [0, vertex_count, instance_count, first_vertex, 0, first_instance, 0, 0];
            }
            metal2vulkan::vk::DrawCommand::DrawIndexed {
                index_type, index_buffer_offset, index_count, instance_count, first_index,
                vertex_offset, first_instance,
            } => {
                out.draw = [1, index_count, instance_count, first_index, vertex_offset as u32,
                            first_instance, raw!(IndexType, index_type), 0];
                out.index_offset = index_buffer_offset;
            }
            _ => return Err("not a direct draw".into()),
        }
        Ok(Box::into_raw(Box::new(out)))
    });
    r.unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_draw_writes(d: *const M2vDraw, n: *mut u32) -> *const LagfxM2vWrite {
    let Some(d) = (unsafe { d.as_ref() }) else { return std::ptr::null() };
    if !n.is_null() { unsafe { *n = d.writes.len() as u32 } }
    d.writes.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_draw_bytes(d: *const M2vDraw, i: u32, len: *mut usize) -> *const u8 {
    let Some(b) = (unsafe { d.as_ref() }).and_then(|d| d.blobs.get(i as usize)) else { return std::ptr::null() };
    if !len.is_null() { unsafe { *len = b.len() } }
    b.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_draw_push(d: *const M2vDraw, n: *mut u32, data: *mut *const u8) -> *const LagfxM2vPush {
    let Some(d) = (unsafe { d.as_ref() }) else { return std::ptr::null() };
    unsafe {
        if !n.is_null() { *n = d.push.len() as u32 }
        if !data.is_null() { *data = d.push_data.as_ptr() }
    }
    d.push.as_ptr()
}

/// Specialization: data bytes and entries {constant id, offset, size}.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_draw_specialization(
    d: *const M2vDraw, data: *mut *const u8, data_len: *mut usize, n_entries: *mut u32,
) -> *const [u32; 3] {
    let Some(d) = (unsafe { d.as_ref() }) else { return std::ptr::null() };
    unsafe {
        if !data.is_null() { *data = d.spec_data.as_ptr() }
        if !data_len.is_null() { *data_len = d.spec_data.len() }
        if !n_entries.is_null() { *n_entries = d.spec_entries.len() as u32 }
    }
    d.spec_entries.as_ptr()
}

/// Vertex buffers {binding, Metal buffer index, offset}.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_draw_vertex_buffers(d: *const M2vDraw, n: *mut u32) -> *const [u64; 3] {
    let Some(d) = (unsafe { d.as_ref() }) else { return std::ptr::null() };
    if !n.is_null() { unsafe { *n = d.vertex_buffers.len() as u32 } }
    d.vertex_buffers.as_ptr()
}

/// The draw: {kind 0 draw / 1 indexed, count, instance_count, first vertex or index,
/// vertex offset, first_instance, VkIndexType, 0}; *index_offset set for indexed.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_draw_call(d: *const M2vDraw, index_offset: *mut u64) -> *const u32 {
    let Some(d) = (unsafe { d.as_ref() }) else { return std::ptr::null() };
    if !index_offset.is_null() { unsafe { *index_offset = d.index_offset } }
    d.draw.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_draw_free(d: *mut M2vDraw) {
    if !d.is_null() {
        drop(unsafe { Box::from_raw(d) });
    }
}

/// The pipeline a draw was planned on (the one given, or a variant of it that
/// lives as long as the original): its SPIR-V and layout are the ones to use.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_draw_pipeline(d: *const M2vDraw) -> *const M2vPipeline {
    unsafe { d.as_ref() }.map_or(std::ptr::null(), |d| d.pipeline)
}

/// Push holes {kind 1 root table / 2 execution status, VkShaderStageFlags, push offset}.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_draw_holes(d: *const M2vDraw, n: *mut u32) -> *const [u32; 3] {
    let Some(d) = (unsafe { d.as_ref() }) else { return std::ptr::null() };
    if !n.is_null() { unsafe { *n = d.holes.len() as u32 } }
    d.holes.as_ptr()
}

/// Root table i: its bytes and stages; NULL past the last. The consumer writes
/// the sink buffer's device address at `lagfx_m2v_root_sink_offset()`.
#[no_mangle]
pub unsafe extern "C" fn lagfx_m2v_draw_root_table(
    d: *const M2vDraw, i: u32, stages: *mut u32, len: *mut usize,
) -> *const u8 {
    let Some((st, b)) = (unsafe { d.as_ref() }).and_then(|d| d.root_tables.get(i as usize)) else {
        return std::ptr::null();
    };
    unsafe {
        if !stages.is_null() { *stages = *st }
        if !len.is_null() { *len = b.len() }
    }
    b.as_ptr()
}

#[no_mangle]
pub extern "C" fn lagfx_m2v_root_sink_offset() -> u32 {
    metal2vulkan::bind::RootTable::SINK_HOST_OFFSET
}
