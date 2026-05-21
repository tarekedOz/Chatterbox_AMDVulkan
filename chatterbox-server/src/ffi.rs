//! Hand-written FFI bindings for chatterbox-cpp/src/chatterbox_capi.h.
//!
//! The C API is intentionally small (8 functions), so hand-rolling is
//! cheaper than depending on bindgen + libclang. Keep in sync with
//! that header.

#![allow(non_camel_case_types)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct chatterbox_ctx {
    _unused: [u8; 0],
}

/// Mirrors chatterbox_gen_params_t in chatterbox_capi.h. Each field uses
/// a sentinel meaning "engine default": negative for the float/int knobs,
/// 0 for seed/max_tokens. `Default` sets those sentinels.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GenParams {
    pub seed: u64,
    pub max_tokens: c_int,
    pub temperature: f32,
    pub top_k: c_int,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub cfm_timesteps: c_int,
    pub exaggeration: f32, // ignored — unsupported by Turbo
    pub cfg_weight: f32,   // ignored — unsupported by Turbo
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams {
            seed: 0,
            max_tokens: 0,
            temperature: -1.0,
            top_k: -1,
            top_p: -1.0,
            repetition_penalty: -1.0,
            cfm_timesteps: -1,
            exaggeration: -1.0,
            cfg_weight: -1.0,
        }
    }
}

#[link(name = "chatterbox", kind = "static")]
extern "C" {
    pub fn chatterbox_init(
        t3_path: *const c_char,
        ve_path: *const c_char,
        s3gen_path: *const c_char,
    ) -> *mut chatterbox_ctx;

    pub fn chatterbox_free(ctx: *mut chatterbox_ctx);

    pub fn chatterbox_load_voices(ctx: *mut chatterbox_ctx, path: *const c_char) -> c_int;
    pub fn chatterbox_voice_count(ctx: *const chatterbox_ctx) -> c_int;
    pub fn chatterbox_voice_name(
        ctx: *const chatterbox_ctx,
        index: c_int,
        out: *mut c_char,
        max_len: c_int,
    ) -> c_int;
    pub fn chatterbox_set_voice(ctx: *mut chatterbox_ctx, name: *const c_char) -> c_int;

    pub fn chatterbox_condition_pcm(
        ctx: *mut chatterbox_ctx,
        samples: *const f32,
        n_samples: c_int,
        sr: c_int,
    ) -> c_int;

    pub fn chatterbox_synthesize_ex(
        ctx: *mut chatterbox_ctx,
        text: *const c_char,
        params: *const GenParams,
        out_pcm: *mut f32,
        max_samples: c_int,
        out_sample_rate: *mut c_int,
    ) -> c_int;

    pub fn chatterbox_last_error() -> *const c_char;
}

/// Safe Rust wrapper around the C engine.
///
/// The engine is NOT thread-safe; this wrapper does not enforce
/// serialization itself — wrap it in a Mutex if you need concurrent
/// synthesize calls.
pub struct Engine {
    ctx: *mut chatterbox_ctx,
}

unsafe impl Send for Engine {}

#[derive(Debug)]
pub enum Error {
    NullArgument,
    Engine(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NullArgument => write!(f, "null FFI argument"),
            Error::Engine(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for Error {}

fn last_error() -> String {
    unsafe {
        let p = chatterbox_last_error();
        if p.is_null() {
            "chatterbox engine error (no message)".into()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

impl Engine {
    pub fn load(
        t3_gguf: &str,
        ve_gguf: &str,
        s3gen_gguf: &str,
    ) -> Result<Self, Error> {
        let t3 = CString::new(t3_gguf).map_err(|_| Error::NullArgument)?;
        let ve = CString::new(ve_gguf).map_err(|_| Error::NullArgument)?;
        let s3gen = CString::new(s3gen_gguf).map_err(|_| Error::NullArgument)?;
        let ctx = unsafe { chatterbox_init(t3.as_ptr(), ve.as_ptr(), s3gen.as_ptr()) };
        if ctx.is_null() {
            return Err(Error::Engine(last_error()));
        }
        Ok(Engine { ctx })
    }

    pub fn load_voices(&mut self, path: &str) -> Result<(), Error> {
        let p = CString::new(path).map_err(|_| Error::NullArgument)?;
        let rc = unsafe { chatterbox_load_voices(self.ctx, p.as_ptr()) };
        if rc != 0 {
            return Err(Error::Engine(last_error()));
        }
        Ok(())
    }

    pub fn voice_names(&self) -> Vec<String> {
        let n = unsafe { chatterbox_voice_count(self.ctx) };
        if n <= 0 {
            return vec![];
        }
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n {
            let mut buf = vec![0i8; 128];
            let written = unsafe {
                chatterbox_voice_name(
                    self.ctx,
                    i,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as c_int,
                )
            };
            if written < 0 {
                continue;
            }
            let cs =
                unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
            out.push(cs.to_string_lossy().into_owned());
        }
        out
    }

    pub fn set_voice(&mut self, name: &str) -> Result<(), Error> {
        let n = CString::new(name).map_err(|_| Error::NullArgument)?;
        let rc = unsafe { chatterbox_set_voice(self.ctx, n.as_ptr()) };
        if rc != 0 {
            return Err(Error::Engine(last_error()));
        }
        Ok(())
    }

    /// Clone a voice from mono fp32 PCM at `sr` Hz. Caches the
    /// conditioning so subsequent synthesize calls use this voice.
    pub fn condition_pcm(&mut self, samples: &[f32], sr: i32) -> Result<(), Error> {
        if samples.is_empty() {
            return Err(Error::Engine("empty reference audio".into()));
        }
        let rc = unsafe {
            chatterbox_condition_pcm(
                self.ctx,
                samples.as_ptr(),
                samples.len() as c_int,
                sr as c_int,
            )
        };
        if rc != 0 {
            return Err(Error::Engine(last_error()));
        }
        Ok(())
    }

    /// Synthesize `text` with per-request params and return
    /// (sample_rate, pcm_fp32_mono). Sentinel fields in `params` fall
    /// back to the engine's load-time defaults.
    pub fn synthesize_ex(
        &mut self,
        text: &str,
        params: &GenParams,
        max_samples: usize,
    ) -> Result<(u32, Vec<f32>), Error> {
        let t = CString::new(text).map_err(|_| Error::NullArgument)?;
        let mut buf = vec![0.0f32; max_samples];
        let mut sr: c_int = 0;
        let n = unsafe {
            chatterbox_synthesize_ex(
                self.ctx,
                t.as_ptr(),
                params as *const GenParams,
                buf.as_mut_ptr(),
                buf.len() as c_int,
                &mut sr,
            )
        };
        if n < 0 {
            return Err(Error::Engine(last_error()));
        }
        buf.truncate(n as usize);
        Ok((sr as u32, buf))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { chatterbox_free(self.ctx) };
    }
}
