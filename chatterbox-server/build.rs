// Locate the C++ chatterbox static library and tell cargo to link it.
//
// We expect the caller to have built chatterbox-cpp via cmake into
// chatterbox-cpp/build/ (CPU-only) — this build script does NOT invoke
// cmake itself. That keeps the Rust build fast and lets us experiment
// with Vulkan / debug builds independently.
//
// Required env vars on Windows (optional override):
//   CHATTERBOX_CPP_BUILD_DIR  -- defaults to ../chatterbox-cpp/build
//
// Links:
//   chatterbox.a         (engine + C API)
//   ggml.a + ggml-base.a + ggml-cpu.a
//   stdc++, pthread, dl, m (Linux); -lstdc++ on Windows

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let build_dir = match env::var("CHATTERBOX_CPP_BUILD_DIR") {
        Ok(p) => PathBuf::from(p),
        Err(_) => manifest_dir
            .parent()
            .unwrap()
            .join("chatterbox-cpp")
            .join("build"),
    };

    // On the scoop-mingw toolchain, libgomp lives next to the compiler
    // under <gcc>/lib. Add it to the linker search path. Resolution
    // order:
    //   1. CHATTERBOX_GCC_LIB_DIR env var (explicit, most reliable —
    //      gcc is often not on the persistent PATH on this setup).
    //   2. `where gcc` probe -> <gcc>/../lib (best effort).
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rerun-if-env-changed=CHATTERBOX_GCC_LIB_DIR");
        let mut lib_dir: Option<PathBuf> = env::var("CHATTERBOX_GCC_LIB_DIR")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.exists());

        if lib_dir.is_none() {
            if let Ok(out) = std::process::Command::new("cmd")
                .args(["/c", "where", "gcc"])
                .output()
            {
                if let Ok(s) = String::from_utf8(out.stdout) {
                    if let Some(gcc) = s.lines().next() {
                        let gcc_path = PathBuf::from(gcc.trim());
                        if let Some(lib) =
                            gcc_path.parent().and_then(|b| b.parent()).map(|r| r.join("lib"))
                        {
                            if lib.exists() {
                                lib_dir = Some(lib);
                            }
                        }
                    }
                }
            }
        }

        if let Some(lib) = lib_dir {
            println!("cargo:rustc-link-search=native={}", lib.display());
        }
    }

    if !build_dir.exists() {
        panic!(
            "chatterbox-cpp build directory not found at {}.\n\
             Build the C++ side first:\n\
                 cd ../chatterbox-cpp\n\
                 cmake -B build -G Ninja\n\
                 cmake --build build",
            build_dir.display()
        );
    }

    let ggml_dir = build_dir.join("third_party").join("ggml").join("src");

    // Library search paths.
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search=native={}", ggml_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        ggml_dir.join("ggml-cpu").display()
    );

    // Static libs (order matters for the GCC linker: dependents first).
    //
    // ggml's sub-build emits filenames that differ by platform:
    //   Linux/macOS: libggml.a, libggml-cpu.a, libggml-base.a
    //   Windows/mingw: ggml.a, ggml-cpu.a, ggml-base.a (no `lib` prefix)
    // On Windows we need the `+verbatim` modifier to pass the file name
    // through unchanged; on Linux the normal `-lggml` works.
    println!("cargo:rustc-link-lib=static=chatterbox");
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=static:+verbatim=ggml.a");
        println!("cargo:rustc-link-lib=static:+verbatim=ggml-cpu.a");
        println!("cargo:rustc-link-lib=static:+verbatim=ggml-base.a");
    } else {
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=static=ggml-base");
    }

    // C++ runtime + OpenMP + system libs. ggml-cpu uses OpenMP for
    // intra-op parallelism, so -lgomp is required when building with
    // GCC.
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=gomp");
        // ggml-cpu uses Windows registry APIs for CPU detection /
        // power throttling — link advapi32 to satisfy them.
        println!("cargo:rustc-link-lib=advapi32");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=m");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=omp");
    }

    println!("cargo:rerun-if-env-changed=CHATTERBOX_CPP_BUILD_DIR");
    println!("cargo:rerun-if-changed=build.rs");
}
