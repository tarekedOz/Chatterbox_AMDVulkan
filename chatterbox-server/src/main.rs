//! Chatterbox HTTP server.
//!
//! Serves the OpenAI /v1/audio/speech contract from a local C++
//! chatterbox engine via FFI (see ffi.rs / chatterbox_capi.h).
//!
//! Configuration is CLI-driven:
//!     chatterbox-server \
//!         --t3-gguf models/chatterbox-turbo-t3-fp16.gguf  \
//!         --ve-gguf models/chatterbox-turbo-ve-fp16.gguf  \
//!         --s3gen-gguf models/chatterbox-turbo-s3gen-fp16.gguf \
//!         --voices-gguf voices.gguf \
//!         --addr 127.0.0.1:8087

mod audio;
mod ffi;

use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Multipart, State},
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use std::convert::Infallible;
use tokio_stream::wrappers::ReceiverStream;
use clap::Parser;
use parking_lot::Mutex;
use rust_embed::RustEmbed;
use serde::Deserialize;
use std::sync::Arc;
use tracing::{error, info};

/// Web UI assets, embedded into the binary at build time so the server
/// stays a single self-contained artifact (no runtime asset directory).
#[derive(RustEmbed)]
#[folder = "web/"]
struct WebAssets;

fn mime_for(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json") | Some("map") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        _ => "application/octet-stream",
    }
}

fn serve_index() -> Response {
    match WebAssets::get("index.html") {
        Some(index) => (
            [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
            index.data.into_owned(),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "UI not bundled").into_response(),
    }
}

/// Fallback handler: serve an embedded asset by request path, or fall
/// back to index.html (SPA routing). Explicit routes (/health,
/// /v1/audio/*, /api/*) take precedence over this.
async fn static_handler(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    if path.is_empty() {
        return serve_index();
    }
    match WebAssets::get(path) {
        Some(content) => (
            [(header::CONTENT_TYPE, mime_for(path))],
            content.data.into_owned(),
        )
            .into_response(),
        None => serve_index(),
    }
}

const DEFAULT_MAX_SAMPLES: usize = 24000 * 30; // 30 seconds @ 24 kHz cap

// Settings resolve with precedence: CLI arg > env var > config file >
// built-in default. CLI/env are handled by clap (env = "..."); the YAML
// config file fills any gaps, and Resolved applies the final defaults.
#[derive(Parser, Debug, Clone)]
#[command(name = "chatterbox-server", about = "OpenAI-compatible TTS server")]
struct Args {
    /// Path to a YAML config file (else ./config.yaml if present).
    #[arg(long, env = "CHATTERBOX_CONFIG")]
    config: Option<String>,
    /// Path to chatterbox_t3 GGUF.
    #[arg(long, env = "CHATTERBOX_T3_GGUF")]
    t3_gguf: Option<String>,
    /// Path to chatterbox_ve GGUF.
    #[arg(long, env = "CHATTERBOX_VE_GGUF")]
    ve_gguf: Option<String>,
    /// Path to chatterbox_s3gen GGUF.
    #[arg(long, env = "CHATTERBOX_S3GEN_GGUF")]
    s3gen_gguf: Option<String>,
    /// Path to voices.gguf (VoicePack).
    #[arg(long, env = "CHATTERBOX_VOICES_GGUF")]
    voices_gguf: Option<String>,
    /// Bind address.
    #[arg(long, env = "CHATTERBOX_ADDR")]
    addr: Option<String>,
    /// Cap on output audio length (samples @ 24 kHz). Default: 30 sec.
    #[arg(long, env = "CHATTERBOX_MAX_SAMPLES")]
    max_samples: Option<usize>,
    /// Max characters per synthesis chunk. Longer input is split on
    /// sentence boundaries and the segments are concatenated.
    #[arg(long, env = "CHATTERBOX_MAX_CHUNK_CHARS")]
    max_chunk_chars: Option<usize>,
}

/// YAML config file schema. All fields optional; CLI/env take precedence.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct FileConfig {
    t3_gguf: Option<String>,
    ve_gguf: Option<String>,
    s3gen_gguf: Option<String>,
    voices_gguf: Option<String>,
    addr: Option<String>,
    max_samples: Option<usize>,
    max_chunk_chars: Option<usize>,
}

struct AppState {
    engine: Mutex<ffi::Engine>,
    max_samples: usize,
    max_chunk_chars: usize,
    voices: Vec<String>,
}

/// OpenAI /v1/audio/speech request body (subset; we accept the keys we
/// understand and ignore the rest).
#[derive(Debug, Deserialize)]
struct SpeechRequest {
    /// Model name, e.g. "tts-1" or "chatterbox-turbo". Currently ignored
    /// (we always run the loaded engine).
    #[allow(dead_code)]
    model: Option<String>,
    /// Text to synthesize. Required.
    input: String,
    /// Voice name (must be in the loaded voice pack).
    voice: String,
    /// "wav" | "pcm" | "mp3". Default "wav". "mp3" is not yet supported.
    #[serde(default)]
    response_format: Option<String>,
    /// 0.25..4.0; speed of speech. Currently ignored (chatterbox has
    /// no straightforward speed control without re-synthesis).
    #[allow(dead_code)]
    #[serde(default)]
    speed: Option<f32>,
    /// Optional seed for deterministic generation.
    #[serde(default)]
    seed: Option<u64>,
}

async fn list_voices(State(app): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"voices": app.voices.clone()}))
}

/// GET /api/config — capabilities the UI needs to render correctly
/// (e.g. which output formats this build supports).
async fn config(State(app): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "formats": audio::available_formats(),
        "voices": app.voices.clone(),
        "max_chunk_chars": app.max_chunk_chars,
    }))
}

async fn health() -> &'static str {
    "ok"
}

/// /api/tts request body (web UI). Like SpeechRequest but uses `text`
/// and `format` (the UI's field names), plus optional per-request
/// generation params. Unset params fall back to engine defaults.
#[derive(Debug, Deserialize)]
struct TtsRequest {
    text: String,
    voice: String,
    #[serde(default)]
    seed: Option<u64>,
    /// "wav" only in Phase 1.
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    cfm_timesteps: Option<i32>,
    /// Stream raw 16-bit PCM (audio/L16) chunk-by-chunk as it is
    /// synthesized, instead of returning a finished file.
    #[serde(default)]
    stream: Option<bool>,
}

fn gen_params_from(req: &TtsRequest) -> ffi::GenParams {
    let mut p = ffi::GenParams {
        seed: req.seed.unwrap_or(0),
        ..Default::default()
    };
    if let Some(v) = req.temperature {
        p.temperature = v;
    }
    if let Some(v) = req.top_p {
        p.top_p = v;
    }
    if let Some(v) = req.top_k {
        p.top_k = v;
    }
    if let Some(v) = req.repetition_penalty {
        p.repetition_penalty = v;
    }
    if let Some(v) = req.cfm_timesteps {
        p.cfm_timesteps = v;
    }
    p
}

fn pcm_to_le_i16(pcm: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for v in pcm {
        let s = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

/// Streaming synthesis: raw 16-bit PCM (audio/L16), one HTTP chunk per
/// synthesized text segment, so playback can start before the whole
/// utterance is done. Format/container options don't apply (containers
/// can't be streamed cleanly); always PCM.
fn tts_stream(app: Arc<AppState>, req: TtsRequest) -> Response {
    let voice = req.voice.clone();
    if !voice.is_empty() && !app.voices.contains(&voice) {
        return (StatusCode::BAD_REQUEST, format!("unknown voice: {voice}")).into_response();
    }
    let params = gen_params_from(&req);
    let chunks = chunk_text(&req.text, app.max_chunk_chars);

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(4);
    let app2 = app.clone();
    tokio::task::spawn_blocking(move || {
        let mut eng = app2.engine.lock();
        if !voice.is_empty() && eng.set_voice(&voice).is_err() {
            return; // ends the stream early
        }
        for chunk in &chunks {
            match eng.synthesize_ex(chunk, &params, app2.max_samples) {
                Ok((_sr, pcm)) => {
                    if tx
                        .blocking_send(Ok(Bytes::from(pcm_to_le_i16(&pcm))))
                        .is_err()
                    {
                        break; // client disconnected
                    }
                }
                Err(e) => {
                    error!("stream synthesize failed: {e}");
                    break;
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/L16")
        .header("x-sample-rate", "24000")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

/// POST /api/tts — the rich web-UI endpoint. Shares the engine + encode
/// path with /v1/audio/speech; distinct field names and WAV-only.
async fn tts(State(app): State<Arc<AppState>>, Json(req): Json<TtsRequest>) -> Response {
    if req.stream.unwrap_or(false) {
        return tts_stream(app, req);
    }
    let fmt = req.format.as_deref().unwrap_or("wav").to_lowercase();
    if !audio::is_available(&fmt) {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported format: {fmt}; available: {}",
                audio::available_formats().join(", ")
            ),
        )
            .into_response();
    }
    let p = gen_params_from(&req);
    synthesize_response(&app, &req.voice, &req.text, p, &fmt)
}

/// Decode a WAV byte blob to mono fp32 PCM + sample rate. Downmixes
/// multichannel by averaging.
fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, i32), String> {
    let cursor = std::io::Cursor::new(bytes);
    let mut reader =
        hound::WavReader::new(cursor).map_err(|e| format!("not a valid WAV: {e}"))?;
    let spec = reader.spec();
    let ch = spec.channels.max(1) as usize;
    let sr = spec.sample_rate as i32;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
        }
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / scale)
                .collect()
        }
    };
    if raw.is_empty() {
        return Err("WAV contained no samples".into());
    }
    if ch <= 1 {
        return Ok((raw, sr));
    }
    let frames = raw.len() / ch;
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..ch {
            acc += raw[f * ch + c];
        }
        mono.push(acc / ch as f32);
    }
    Ok((mono, sr))
}

/// POST /api/clone — multipart upload of a reference WAV. Decodes,
/// conditions the engine on it, and leaves it as the active voice.
async fn clone_voice(State(app): State<Arc<AppState>>, mut mp: Multipart) -> Response {
    let mut audio: Option<Vec<u8>> = None;
    loop {
        match mp.next_field().await {
            Ok(Some(field)) => match field.bytes().await {
                Ok(b) => audio = Some(b.to_vec()),
                Err(e) => {
                    return (StatusCode::BAD_REQUEST, format!("upload read error: {e}"))
                        .into_response();
                }
            },
            Ok(None) => break,
            Err(e) => {
                return (StatusCode::BAD_REQUEST, format!("multipart error: {e}"))
                    .into_response();
            }
        }
    }
    let Some(bytes) = audio else {
        return (StatusCode::BAD_REQUEST, "no audio file in upload").into_response();
    };
    let (pcm, sr) = match decode_wav(&bytes) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };
    {
        let mut eng = app.engine.lock();
        if let Err(e) = eng.condition_pcm(&pcm, sr) {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("condition: {e}"))
                .into_response();
        }
    }
    info!(samples = pcm.len(), sample_rate = sr, "cloned voice from upload");
    Json(serde_json::json!({"status": "ok", "samples": pcm.len(), "sample_rate": sr}))
        .into_response()
}

/// POST /v1/audio/speech
async fn speech(
    State(app): State<Arc<AppState>>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    let fmt = req
        .response_format
        .as_deref()
        .unwrap_or("wav")
        .to_lowercase();
    if !audio::is_available(&fmt) {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported response_format: {fmt}; available: {}",
                audio::available_formats().join(", ")
            ),
        )
            .into_response();
    }
    // OpenAI shape carries no sampling knobs; pass seed only, rest default.
    let p = ffi::GenParams {
        seed: req.seed.unwrap_or(0),
        ..Default::default()
    };
    synthesize_response(&app, &req.voice, &req.input, p, &fmt)
}

/// Split text into sentences on . ! ? and newlines, keeping the
/// trailing punctuation with its sentence.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            let t = cur.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
            cur.clear();
        }
    }
    let t = cur.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    out
}

/// Greedily pack `s`'s words into pieces of at most `max_chars`.
fn hard_split(s: &str, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        if cur.is_empty() {
            cur = word.to_string();
        } else if cur.len() + 1 + word.len() <= max_chars {
            cur.push(' ');
            cur.push_str(word);
        } else {
            out.push(std::mem::take(&mut cur));
            cur = word.to_string();
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Split `text` into synthesis chunks of at most `max_chars`, breaking
/// on sentence boundaries (and on words for over-long sentences).
fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for s in split_sentences(text) {
        if s.len() > max_chars {
            if !cur.is_empty() {
                chunks.push(std::mem::take(&mut cur));
            }
            chunks.extend(hard_split(&s, max_chars));
        } else if cur.is_empty() {
            cur = s;
        } else if cur.len() + 1 + s.len() <= max_chars {
            cur.push(' ');
            cur.push_str(&s);
        } else {
            chunks.push(std::mem::take(&mut cur));
            cur = s;
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    if chunks.is_empty() {
        chunks.push(text.trim().to_string());
    }
    chunks
}

/// Shared synthesis + encode path. `fmt` is assumed already validated
/// ("wav" or "pcm"). Locks the engine mutex (engine is NOT thread-safe).
/// Long text is split into chunks, synthesized in turn, and concatenated
/// with a short silence between segments.
fn synthesize_response(
    app: &AppState,
    voice: &str,
    text: &str,
    params: ffi::GenParams,
    fmt: &str,
) -> Response {
    let chunks = chunk_text(text, app.max_chunk_chars);
    let (sr, pcm) = {
        let mut eng = app.engine.lock();
        // Empty voice = use whatever conditioning is already active
        // (e.g. a clone from /api/clone). A named voice (re)sets it.
        // Set once; conditioning persists across the chunk loop.
        if !voice.is_empty() {
            if let Err(e) = eng.set_voice(voice) {
                return (StatusCode::BAD_REQUEST, format!("set_voice: {e}")).into_response();
            }
        }
        let mut sr = 0u32;
        let mut pcm: Vec<f32> = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            match eng.synthesize_ex(chunk, &params, app.max_samples) {
                Ok((csr, cpcm)) => {
                    sr = csr;
                    if i > 0 {
                        // ~120 ms of silence between segments.
                        let gap = (csr as f32 * 0.12) as usize;
                        pcm.resize(pcm.len() + gap, 0.0);
                    }
                    pcm.extend_from_slice(&cpcm);
                }
                Err(e) => {
                    error!("synthesize failed (chunk {i}): {e}");
                    return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response();
                }
            }
        }
        (sr, pcm)
    };

    info!(
        samples = pcm.len(),
        sample_rate = sr,
        chunks = chunks.len(),
        "synthesized"
    );

    let (bytes, ctype) = match audio::encode(fmt, &pcm, sr) {
        Ok(v) => v,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    };
    // PCM carries the sample rate in a header (it has no container).
    if fmt == "pcm" {
        let sr_hdr: &'static str =
            Box::leak(format!("{sr}").into_boxed_str());
        return (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, ctype),
                (header::HeaderName::from_static("x-sample-rate"), sr_hdr),
            ],
            Bytes::from(bytes),
        )
            .into_response();
    }
    (StatusCode::OK, [(header::CONTENT_TYPE, ctype)], Bytes::from(bytes)).into_response()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "chatterbox_server=info,tower_http=info".into()),
        )
        .init();

    let args = Args::parse();

    // Load the YAML config file (--config / CHATTERBOX_CONFIG, else
    // ./config.yaml if present). CLI/env values take precedence below.
    let cfg_path = args.config.clone().or_else(|| {
        std::path::Path::new("config.yaml")
            .exists()
            .then(|| "config.yaml".to_string())
    });
    let file_cfg: FileConfig = match &cfg_path {
        Some(p) => {
            let text = std::fs::read_to_string(p)
                .map_err(|e| format!("reading config {p}: {e}"))?;
            info!(config = %p, "loaded config file");
            serde_yaml::from_str(&text).map_err(|e| format!("parsing config {p}: {e}"))?
        }
        None => FileConfig::default(),
    };

    // Resolve: CLI/env (args) > config file > built-in default.
    let need = |cli: Option<String>, file: Option<String>, name: &str| {
        cli.or(file)
            .ok_or_else(|| format!("{name} is required (pass --{name}, set its env var, or put it in the config file)"))
    };
    let t3_gguf = need(args.t3_gguf, file_cfg.t3_gguf, "t3-gguf")?;
    let ve_gguf = need(args.ve_gguf, file_cfg.ve_gguf, "ve-gguf")?;
    let s3gen_gguf = need(args.s3gen_gguf, file_cfg.s3gen_gguf, "s3gen-gguf")?;
    let voices_gguf = need(args.voices_gguf, file_cfg.voices_gguf, "voices-gguf")?;
    let addr = args
        .addr
        .or(file_cfg.addr)
        .unwrap_or_else(|| "127.0.0.1:8087".to_string());
    let max_samples = args
        .max_samples
        .or(file_cfg.max_samples)
        .unwrap_or(DEFAULT_MAX_SAMPLES);
    let max_chunk_chars = args
        .max_chunk_chars
        .or(file_cfg.max_chunk_chars)
        .unwrap_or(300);

    info!(t3=%t3_gguf, ve=%ve_gguf, s3gen=%s3gen_gguf,
          voices=%voices_gguf, "loading models");

    let mut engine = ffi::Engine::load(&t3_gguf, &ve_gguf, &s3gen_gguf)?;
    engine.load_voices(&voices_gguf)?;
    let voices = engine.voice_names();
    info!(?voices, "voice pack loaded");

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        max_samples,
        max_chunk_chars,
        voices,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/audio/voices", get(list_voices))
        .route("/v1/audio/speech", post(speech))
        // Rich /api/* namespace used by the web UI (distinct from the
        // frozen OpenAI /v1/audio/* routes above).
        .route("/api/voices", get(list_voices))
        .route("/api/config", get(config))
        .route("/api/tts", post(tts))
        // Voice cloning: up to 32 MB reference upload.
        .route(
            "/api/clone",
            post(clone_voice).layer(DefaultBodyLimit::max(32 * 1024 * 1024)),
        )
        // Embedded web UI: any unmatched route serves a static asset or
        // falls back to index.html. Keeps /v1/*, /api/*, /health
        // authoritative.
        .fallback(static_handler)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
