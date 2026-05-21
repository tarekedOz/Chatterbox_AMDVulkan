//! Output-format encoders. WAV and raw PCM are always available; MP3 and
//! Opus are compiled in only with the `audio-formats` cargo feature
//! (they pull in C-library encoders), so the lean build stays small.

/// Formats this build can produce, in UI display order.
pub fn available_formats() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut v = vec!["wav", "pcm"];
    #[cfg(feature = "audio-formats")]
    {
        v.push("mp3");
        v.push("opus");
    }
    v
}

pub fn is_available(fmt: &str) -> bool {
    available_formats().contains(&fmt)
}

pub fn content_type(fmt: &str) -> &'static str {
    match fmt {
        "wav" => "audio/wav",
        "pcm" => "audio/L16",
        "mp3" => "audio/mpeg",
        "opus" => "audio/ogg",
        _ => "application/octet-stream",
    }
}

/// f32 [-1,1] mono -> interleaved little-endian i16.
fn to_i16(pcm: &[f32]) -> Vec<i16> {
    pcm.iter()
        .map(|v| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16)
        .collect()
}

/// Raw little-endian 16-bit PCM (no header).
pub fn encode_pcm(pcm: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for s in to_i16(pcm) {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

/// 16-bit mono WAV.
pub fn encode_wav(pcm: &[f32], sr: u32) -> Result<Vec<u8>, String> {
    let mut buf = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    {
        let mut w = hound::WavWriter::new(&mut buf, spec).map_err(|e| e.to_string())?;
        for s in to_i16(pcm) {
            w.write_sample(s).map_err(|e| e.to_string())?;
        }
        w.finalize().map_err(|e| e.to_string())?;
    }
    Ok(buf.into_inner())
}

#[cfg(feature = "audio-formats")]
pub fn encode_mp3(pcm: &[f32], sr: u32) -> Result<Vec<u8>, String> {
    use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, MonoPcm, Quality};
    let mut builder = Builder::new().ok_or("mp3: LAME init failed")?;
    builder.set_num_channels(1).map_err(|e| format!("mp3: {e:?}"))?;
    builder.set_sample_rate(sr).map_err(|e| format!("mp3: {e:?}"))?;
    builder
        .set_brate(Bitrate::Kbps128)
        .map_err(|e| format!("mp3: {e:?}"))?;
    builder
        .set_quality(Quality::Good)
        .map_err(|e| format!("mp3: {e:?}"))?;
    let mut enc = builder.build().map_err(|e| format!("mp3: {e:?}"))?;

    let samples = to_i16(pcm);
    let mut out: Vec<u8> = Vec::new();
    out.reserve(mp3lame_encoder::max_required_buffer_size(samples.len()));
    let n = enc
        .encode(MonoPcm(&samples), out.spare_capacity_mut())
        .map_err(|e| format!("mp3 encode: {e:?}"))?;
    unsafe { out.set_len(out.len() + n) };
    let n = enc
        .flush::<FlushNoGap>(out.spare_capacity_mut())
        .map_err(|e| format!("mp3 flush: {e:?}"))?;
    unsafe { out.set_len(out.len() + n) };
    Ok(out)
}

/// Opus in an Ogg container (RFC 7845). Encodes at the input rate (must
/// be a valid Opus rate: 8/12/16/24/48 kHz; chatterbox is 24 kHz).
#[cfg(feature = "audio-formats")]
pub fn encode_opus(pcm: &[f32], sr: u32) -> Result<Vec<u8>, String> {
    use audiopus::{coder::Encoder, Application, Channels, SampleRate};
    use ogg::PacketWriteEndInfo;

    let rate = match sr {
        8000 => SampleRate::Hz8000,
        12000 => SampleRate::Hz12000,
        16000 => SampleRate::Hz16000,
        24000 => SampleRate::Hz24000,
        48000 => SampleRate::Hz48000,
        _ => return Err(format!("opus: unsupported sample rate {sr}")),
    };
    let mut enc = Encoder::new(rate, Channels::Mono, Application::Audio)
        .map_err(|e| format!("opus init: {e}"))?;

    let samples = to_i16(pcm);
    // 20 ms frames.
    let frame = (sr as usize / 50).max(1);
    // Ogg granule positions are always expressed at 48 kHz.
    let gp_scale = 48000u64 / sr as u64;

    let mut ogg_out: Vec<u8> = Vec::new();
    let serial: u32 = 0x0B0B_0B0B;
    {
        let mut w = ogg::PacketWriter::new(&mut ogg_out);

        // OpusHead (BOS).
        let mut head = Vec::with_capacity(19);
        head.extend_from_slice(b"OpusHead");
        head.push(1); // version
        head.push(1); // channels
        head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
        head.extend_from_slice(&sr.to_le_bytes()); // original input rate
        head.extend_from_slice(&0i16.to_le_bytes()); // output gain
        head.push(0); // channel mapping family
        w.write_packet(head, serial, PacketWriteEndInfo::EndPage, 0)
            .map_err(|e| format!("opus ogg head: {e}"))?;

        // OpusTags.
        let vendor = b"chatterbox-server";
        let mut tags = Vec::new();
        tags.extend_from_slice(b"OpusTags");
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor);
        tags.extend_from_slice(&0u32.to_le_bytes()); // user comment count
        w.write_packet(tags, serial, PacketWriteEndInfo::EndPage, 0)
            .map_err(|e| format!("opus ogg tags: {e}"))?;

        let n_frames = (samples.len() + frame - 1) / frame;
        let mut buf = vec![0u8; 4000];
        let mut produced: u64 = 0; // samples encoded at input rate
        for i in 0..n_frames {
            let start = i * frame;
            let end = (start + frame).min(samples.len());
            let mut chunk = samples[start..end].to_vec();
            chunk.resize(frame, 0); // pad final frame with silence
            let len = enc
                .encode(&chunk, &mut buf)
                .map_err(|e| format!("opus encode: {e}"))?;
            produced += frame as u64;
            let last = i + 1 == n_frames;
            let end_info = if last {
                PacketWriteEndInfo::EndStream
            } else {
                PacketWriteEndInfo::NormalPacket
            };
            w.write_packet(buf[..len].to_vec(), serial, end_info, produced * gp_scale)
                .map_err(|e| format!("opus ogg page: {e}"))?;
        }
    }
    Ok(ogg_out)
}

/// Encode PCM to the requested format. Returns (bytes, content_type).
/// `fmt` must be validated against `available_formats()` first.
pub fn encode(fmt: &str, pcm: &[f32], sr: u32) -> Result<(Vec<u8>, &'static str), String> {
    match fmt {
        "wav" => Ok((encode_wav(pcm, sr)?, content_type("wav"))),
        "pcm" => Ok((encode_pcm(pcm), content_type("pcm"))),
        #[cfg(feature = "audio-formats")]
        "mp3" => Ok((encode_mp3(pcm, sr)?, content_type("mp3"))),
        #[cfg(feature = "audio-formats")]
        "opus" => Ok((encode_opus(pcm, sr)?, content_type("opus"))),
        _ => Err(format!("unsupported or unavailable format: {fmt}")),
    }
}
