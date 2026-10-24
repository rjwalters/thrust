// Include the shipped build helper so this test cannot drift into a reimplementation.
#[allow(dead_code, unexpected_cfgs)]
mod upstream_build {
    include!("../../../candidate/torch-sys-0.22.0/build.rs");

    pub fn exercise(input: &std::path::Path, output: &std::path::Path) -> anyhow::Result<()> {
        extract(input, output)
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    upstream_build::exercise(
        std::path::Path::new(&args[1]),
        std::path::Path::new(&args[2]),
    )
}
