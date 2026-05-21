// Chatterbox TTS web UI — Alpine.js component.
//
// Talks to the rich /api/* endpoints (added in later Phase 1 tasks):
//   GET  /api/voices  -> { voices: [name, ...] }
//   POST /api/tts     -> audio bytes (audio/wav)
// The OpenAI /v1/audio/* contract is untouched.

// Engine defaults (mirror ChatterboxConfig). Used for initial state and
// the "Reset to defaults" button.
const GEN_DEFAULTS = {
  temperature: 0.7,
  topP: 0.95,
  topK: 50,
  repetitionPenalty: 1.5,
  cfmTimesteps: 2,
};

function ttsApp() {
  return {
    // --- state ---
    text: "",
    voice: "",
    seed: 0,
    format: "wav",
    formats: ["wav", "pcm"],
    maxChars: 1000,
    voices: [],
    voicesError: "",
    busy: false,
    error: "",
    audioUrl: "",
    downloadName: "chatterbox.wav",
    theme: "dark",
    // voice cloning
    cloneFile: null,
    cloning: false,
    useClone: false,
    cloneStatus: "",
    cloneError: false,
    // presets
    presets: {},
    presetName: "",
    selectedPreset: "",
    // generation history (metadata only; audio blobs are session-scoped)
    history: [],
    // streaming
    stream: false,
    streamSamples: 0,
    // generation params
    ...GEN_DEFAULTS,

    // --- lifecycle ---
    init() {
      // Restore persisted settings.
      this.theme = localStorage.getItem("cb_theme") || "dark";
      document.documentElement.setAttribute("data-theme", this.theme);
      this.presets = JSON.parse(localStorage.getItem("cb_presets") || "{}");
      this.history = JSON.parse(localStorage.getItem("cb_history") || "[]");
      const saved = JSON.parse(localStorage.getItem("cb_settings") || "{}");
      if (typeof saved.voice === "string") this.voice = saved.voice;
      if (typeof saved.seed === "number") this.seed = saved.seed;
      if (typeof saved.format === "string") this.format = saved.format;
      for (const k of Object.keys(GEN_DEFAULTS)) {
        if (typeof saved[k] === "number") this[k] = saved[k];
      }

      // Persist on change.
      for (const k of ["voice", "seed", "format", ...Object.keys(GEN_DEFAULTS)]) {
        this.$watch(k, () => this.saveSettings());
      }

      this.loadConfig();
      this.loadVoices();
    },

    async loadConfig() {
      try {
        const res = await fetch("/api/config");
        if (!res.ok) return;
        const cfg = await res.json();
        if (Array.isArray(cfg.formats) && cfg.formats.length) {
          this.formats = cfg.formats;
          if (!this.formats.includes(this.format)) this.format = this.formats[0];
        }
      } catch (e) {
        console.error("loadConfig:", e);
      }
    },

    saveSettings() {
      const s = { voice: this.voice, seed: this.seed, format: this.format };
      for (const k of Object.keys(GEN_DEFAULTS)) s[k] = this[k];
      localStorage.setItem("cb_settings", JSON.stringify(s));
    },

    resetParams() {
      Object.assign(this, GEN_DEFAULTS);
    },

    // --- presets ---
    get presetFields() {
      return [...Object.keys(GEN_DEFAULTS), "format"];
    },

    persistPresets() {
      localStorage.setItem("cb_presets", JSON.stringify(this.presets));
    },

    savePreset() {
      const name = this.presetName.trim();
      if (!name) return;
      const p = {};
      for (const k of this.presetFields) p[k] = this[k];
      this.presets = { ...this.presets, [name]: p };
      this.persistPresets();
      this.selectedPreset = name;
      this.presetName = "";
    },

    loadPreset(name) {
      const p = this.presets[name];
      if (!p) return;
      for (const k of this.presetFields) {
        if (p[k] !== undefined) this[k] = p[k];
      }
    },

    deletePreset(name) {
      if (!name || !this.presets[name]) return;
      const next = { ...this.presets };
      delete next[name];
      this.presets = next;
      this.persistPresets();
      if (this.selectedPreset === name) this.selectedPreset = "";
    },

    // --- history ---
    persistHistory() {
      localStorage.setItem("cb_history", JSON.stringify(this.history));
    },

    addHistory() {
      const entry = {
        id: Date.now() + "-" + Math.random().toString(36).slice(2, 7),
        ts: Date.now(),
        text: this.text,
        voice: this.voice,
        useClone: this.useClone,
        seed: this.seed,
        format: this.format,
      };
      for (const k of Object.keys(GEN_DEFAULTS)) entry[k] = this[k];
      this.history = [entry, ...this.history].slice(0, 20);
      this.persistHistory();
    },

    replay(h) {
      this.text = h.text;
      this.seed = h.seed || 0;
      this.format = this.formats.includes(h.format) ? h.format : this.format;
      for (const k of Object.keys(GEN_DEFAULTS)) {
        if (h[k] !== undefined) this[k] = h[k];
      }
      // Cloned entries replay against the current server-side conditioning.
      if (!h.useClone && h.voice && this.voices.includes(h.voice)) {
        this.useClone = false;
        this.voice = h.voice;
      }
      this.generate();
    },

    clearHistory() {
      this.history = [];
      this.persistHistory();
    },

    // --- voice cloning ---
    onCloneFile(e) {
      this.cloneFile = (e.target.files && e.target.files[0]) || null;
      this.cloneStatus = "";
      this.cloneError = false;
    },

    async clone() {
      if (!this.cloneFile || this.cloning) return;
      this.cloning = true;
      this.cloneStatus = "";
      this.cloneError = false;
      try {
        const fd = new FormData();
        fd.append("file", this.cloneFile);
        const res = await fetch("/api/clone", { method: "POST", body: fd });
        if (!res.ok) throw new Error((await res.text()) || "HTTP " + res.status);
        const data = await res.json();
        this.useClone = true;
        const secs = data.sample_rate
          ? (data.samples / data.sample_rate).toFixed(1) + "s"
          : data.samples + " samples";
        this.cloneStatus = "Cloned from " + this.cloneFile.name + " (" + secs + ")";
      } catch (e) {
        this.cloneError = true;
        this.cloneStatus = "Clone failed: " + e.message;
        console.error("clone:", e);
      } finally {
        this.cloning = false;
      }
    },

    useLibraryVoice() {
      this.useClone = false;
      this.cloneStatus = "";
      this.cloneError = false;
    },

    toggleTheme() {
      this.theme = this.theme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", this.theme);
      localStorage.setItem("cb_theme", this.theme);
    },

    // --- API ---
    async loadVoices() {
      this.voicesError = "";
      try {
        const res = await fetch("/api/voices");
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        this.voices = Array.isArray(data.voices) ? data.voices : [];
        if (this.voices.length === 0) this.voicesError = "No voices loaded";
        // Keep the persisted choice if still valid, else pick the first.
        if (!this.voices.includes(this.voice)) this.voice = this.voices[0] || "";
      } catch (e) {
        this.voices = [];
        this.voicesError = "Failed to load voices";
        console.error("loadVoices:", e);
      }
    },

    // Request body shared by streamed and non-streamed generation.
    requestBody(extra) {
      return JSON.stringify({
        text: this.text,
        voice: this.useClone ? "" : this.voice,
        seed: this.seed || 0,
        format: this.format,
        temperature: this.temperature,
        top_p: this.topP,
        top_k: this.topK,
        repetition_penalty: this.repetitionPenalty,
        cfm_timesteps: this.cfmTimesteps,
        ...extra,
      });
    },

    // Progressive playback: read the chunked PCM stream and schedule each
    // segment through Web Audio so sound starts before synthesis finishes.
    async generateStream() {
      this.busy = true;
      this.error = "";
      this.streamSamples = 0;
      let ctx;
      try {
        ctx = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: 24000,
        });
        try { await ctx.resume(); } catch (e) { /* autoplay-suspended is fine */ }
        const res = await fetch("/api/tts", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: this.requestBody({ stream: true }),
        });
        if (!res.ok) throw new Error((await res.text()) || "HTTP " + res.status);
        const reader = res.body.getReader();
        let playhead = ctx.currentTime + 0.15;
        let leftover = new Uint8Array(0);
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const buf = new Uint8Array(leftover.length + value.length);
          buf.set(leftover);
          buf.set(value, leftover.length);
          const n = Math.floor(buf.length / 2);
          const dv = new DataView(buf.buffer, buf.byteOffset, n * 2);
          const f32 = new Float32Array(n);
          for (let i = 0; i < n; i++) f32[i] = dv.getInt16(i * 2, true) / 32768;
          leftover = buf.slice(n * 2);
          if (n > 0) {
            const ab = ctx.createBuffer(1, n, 24000);
            ab.getChannelData(0).set(f32);
            const src = ctx.createBufferSource();
            src.buffer = ab;
            src.connect(ctx.destination);
            src.start(playhead);
            playhead += ab.duration;
            this.streamSamples += n;
          }
        }
        if (this.streamSamples > 0) this.addHistory();
      } catch (e) {
        this.error = "Stream failed: " + e.message;
        console.error("generateStream:", e);
      } finally {
        this.busy = false;
      }
    },

    async generate() {
      if (this.busy || !this.text.trim() || (!this.useClone && !this.voice)) return;
      if (this.stream) return this.generateStream();
      this.busy = true;
      this.error = "";
      // Release the previous object URL to avoid leaking blobs.
      if (this.audioUrl) {
        URL.revokeObjectURL(this.audioUrl);
        this.audioUrl = "";
      }
      try {
        const res = await fetch("/api/tts", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: this.requestBody(),
        });
        if (!res.ok) {
          const msg = await res.text();
          throw new Error(msg || "HTTP " + res.status);
        }
        const blob = await res.blob();
        this.audioUrl = URL.createObjectURL(blob);
        const vname = this.useClone ? "cloned" : this.voice.toLowerCase();
        this.downloadName = "chatterbox-" + vname + "." + this.format;
        this.addHistory();
      } catch (e) {
        this.error = "Synthesis failed: " + e.message;
        console.error("generate:", e);
      } finally {
        this.busy = false;
      }
    },
  };
}
