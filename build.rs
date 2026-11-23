use std::env;
use std::path::PathBuf;

fn main() {
    let target_pointer_width = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default();
    if target_pointer_width != "64" {
        println!(
            "cargo:warning=we didn't build this for 32-bit, but if you're stubborn enough to compile it on x86 anyway, we won't stop you :)"
        );
    }

    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    let def_path = manifest_dir.join("exports").join("OmniMIDI.def");
    let map_path = manifest_dir.join("exports").join("libOmniMIDI.map");

    println!("cargo:rerun-if-changed={}", def_path.display());
    println!("cargo:rerun-if-changed={}", map_path.display());

    match target_os.as_str() {
        "windows" => {
            if target_env == "gnu" {
                // Pass the .def file directly to MinGW ld so only the clean undecorated
                // 64-bit KDMAPI symbols are exported from OmniMIDI.dll.
                println!("cargo:rustc-cdylib-link-arg={}", def_path.display());
            } else if target_env == "msvc" {
                println!("cargo:rustc-cdylib-link-arg=/DEF:{}", def_path.display());
            }
        }
        "linux" => {
            // Restrict exported dynamic symbols in libOmniMIDI.so strictly to the KDMAPI ABI
            // to prevent symbol collisions with host players (e.g., Wasabi/BASS).
            println!(
                "cargo:rustc-cdylib-link-arg=-Wl,--version-script={}",
                map_path.display()
            );
        }
        _ => {}
    }
}
