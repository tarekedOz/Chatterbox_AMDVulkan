# Windows installer (thin, download-on-first-run)

The installer ships only the self-contained **~8 MB** `chatterbox-server.exe`
(engine + embedded web UI + HTTP, statically linked, GPU via Vulkan). The
**~1.4 GB model weights are not bundled** — they download + verify on first
launch, the same pattern as Ollama / LM Studio / whisper.cpp.

## Files here

| File | Role |
|---|---|
| `install.ps1`          | **one-call** bootstrap (`irm <url>/install.ps1 \| iex`): download app zip → fetch weights → shortcut → launch |
| `models.manifest.json` | weights to fetch: name, dest, size, sha256, `base_url` |
| `fetch-models.ps1`     | first-run downloader (skip-if-present, sha256-verify, resumable) |
| `launch.ps1`           | shortcut target: fetch-if-missing → start server → open browser |
| `uninstall.ps1`        | target of the Uninstall shortcut: stop server → remove install dir (incl. weights) + shortcuts |
| `config.yaml`          | installed server config (model paths, bind addr) |
| `chatterbox.iss`       | Inno Setup script that produces `chatterbox-tts-setup.exe` |

## Two ways to install

- **One call** (no compiled installer needed): publish `install.ps1` +
  an app zip `chatterbox-tts-win-x64.zip` (server.exe + `fetch-models.ps1`
  + `launch.ps1` + `uninstall.ps1` + `models.manifest.json` + `config.yaml`)
  + the GGUFs as release assets. Users run `irm <base>/install.ps1 | iex`.
  Build the zip:
  ```powershell
  $stage = "$env:TEMP\cbpkg"; ni -ItemType Directory -Force $stage | Out-Null
  copy chatterbox-server\target\release\chatterbox-server.exe $stage\
  copy dist\fetch-models.ps1,dist\launch.ps1,dist\uninstall.ps1,dist\models.manifest.json,dist\config.yaml $stage\
  Compress-Archive "$stage\*" dist\chatterbox-tts-win-x64.zip -Force
  ```
- **GUI installer**: compile `chatterbox.iss` with Inno Setup (below).

## Uninstall / upgrade

**Uninstall**
- *GUI install* (`chatterbox-tts-setup.exe`): **Settings → Apps**, or the
  **Uninstall Chatterbox TTS** Start Menu entry. The `[UninstallDelete]` rule
  removes the downloaded weights too.
- *One-call install*: run the **Uninstall Chatterbox TTS** Start Menu shortcut
  (it runs `uninstall.ps1`), or manually delete `%LOCALAPPDATA%\Chatterbox TTS`
  plus the Start Menu shortcuts.

**Upgrade** — there is no auto-update; re-run the installer against the new
release. The one-call path re-extracts the app (an existing `config.yaml` is
preserved) and `fetch-models.ps1` re-downloads only weights whose SHA-256
changed. The GUI path installs in place over the previous version (the fixed
`AppId` keeps it a single Add/Remove Programs entry).

## How first-run download works

`launch.ps1` calls `fetch-models.ps1`, which reads `models.manifest.json`
and, for each file, downloads `<base_url>/<name>` to `<dest>` unless it's
already present with a matching SHA-256. Verified files are skipped, so a
re-run resumes a partial install:

```powershell
# default: manifest base_url (the public release host)
.\fetch-models.ps1
```

The download host is overridable with `-BaseUrl` (e.g. a mirror), and
`-Token` adds an auth header for private hosts. Public release assets (the
intended GitHub distribution) download anonymously, no token needed.

## Building the installer

1. **Build the Vulkan server binary** and stage it here as
   `chatterbox-server.exe`:
   ```powershell
   $env:CC="$HOME/scoop/apps/gcc/current/bin/gcc.exe"; $env:CXX="$HOME/scoop/apps/gcc/current/bin/g++.exe"
   cmake -S chatterbox-cpp -B chatterbox-cpp/build_vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DCHATTERBOX_VULKAN=ON
   cmake --build chatterbox-cpp/build_vk --target chatterbox
   $env:CHATTERBOX_CPP_BUILD_DIR="$PWD/chatterbox-cpp/build_vk"
   $env:Path="$HOME/scoop/apps/gcc/current/bin;$env:Path"; $env:CHATTERBOX_GCC_LIB_DIR="$HOME/scoop/apps/gcc/current/lib"
   cargo build --release --manifest-path chatterbox-server/Cargo.toml --features audio-formats
   copy chatterbox-server\target\release\chatterbox-server.exe dist\
   ```
   (Omit `--features audio-formats` for a WAV/PCM-only build; the
   audio-formats C-lib encoders need their libs available to the linker.)
2. **Publish the weights** to the release host and set `base_url` in
   `models.manifest.json` to match (e.g.
   `https://github.com/tarekedOz/Chatterbox_AMDVulkan/releases/download/models-v1`).
3. **Compile**: `ISCC.exe chatterbox.iss` (Inno Setup 6) →
   `Output\chatterbox-tts-setup.exe`.

## Publishing the weights

**GitHub (public, anonymous download):** attach the four GGUFs as release
assets under a tag (e.g. `models-v1`); `base_url` is then
`https://github.com/tarekedOz/Chatterbox_AMDVulkan/releases/download/models-v1`.
The downloader verifies each file against the SHA-256 in the manifest, so
the upload just needs to match those hashes.
