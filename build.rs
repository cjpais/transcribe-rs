// whisper.cpp's Vulkan backend requires Windows registry access (via advapi32.lib)
// to correctly identify and initialize graphics drivers on Windows.
fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("windows") {
        println!("cargo:rustc-link-lib=advapi32");
    }
}
