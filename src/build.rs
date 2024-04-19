use std::env;

#[allow(dead_code)]
fn main() {
    if let Ok(target) = env::var("TARGET") {
        if target == "aarch64-apple-darwin" {
            println!("cargo:rustc-cfg=feature=\"metal\"");
        }
    }
}
