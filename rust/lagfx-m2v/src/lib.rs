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
        Ok(Box::into_raw(Box::new(M2vPipeline { pipeline })))
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
