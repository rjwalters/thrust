use candle_core::Tensor;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("missing fixture")?;
    if path.ends_with(".pth") {
        let tensors = candle_core::pickle::PthTensors::new(&path, None)?;
        let tensor = tensors.get("tensor_fortran")?.ok_or("missing PTH tensor")?;
        println!("PTH SHAPE={:?}; VALUES={:?}", tensor.dims(), tensor.flatten_all()?.to_vec1::<i64>()?);
        return Ok(());
    }
    let tensors = Tensor::read_npz(&path)?;
    let index = candle_core::npy::NpzTensors::new(&path)?;
    for (name, tensor) in tensors {
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        println!("NAME={:?}; SHAPE={:?}; VALUES={:?}", name, tensor.dims(), values);
        let indexed = index.get(&name)?.ok_or("index omitted tensor")?;
        assert_eq!(indexed.dims(), tensor.dims());
        assert_eq!(indexed.flatten_all()?.to_vec1::<f32>()?, values);
        let named = Tensor::read_npz_by_name(&path, &[&name])?;
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].flatten_all()?.to_vec1::<f32>()?, values);
    }
    assert!(index.get("definitely-missing")?.is_none());
    Ok(())
}
