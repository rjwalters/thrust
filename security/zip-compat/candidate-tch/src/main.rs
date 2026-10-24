use tch::Tensor;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("missing fixture")?;
    for (name, tensor) in Tensor::read_npz(&path)? {
        let values = Vec::<f32>::try_from(&tensor.flatten(0, -1))?;
        println!("NAME={:?}; SHAPE={:?}; VALUES={:?}", name, tensor.size(), values);
    }
    Ok(())
}
