/// ! Debug script to check if gradients are flowing properly
use anyhow::Result;
use tch::{Kind, Tensor};
use thrust_rl::policy::mlp::MlpPolicy;

fn main() -> Result<()> {
    println!("🔍 Debugging gradient flow...\n");

    // Create policy
    let mut policy = MlpPolicy::new(4, 2, 64);
    let device = policy.device();

    // Create optimizer
    let mut optimizer = policy.optimizer(3e-4);

    println!("Initial network check:");
    let obs = Tensor::randn([8, 4], (Kind::Float, device));
    let (logits_before, values_before) = policy.forward(&obs);
    println!("  Logits shape: {:?}", logits_before.size());
    println!("  Values shape: {:?}", values_before.size());
    println!("  Logits sample: {:?}", Vec::<f32>::try_from(logits_before.get(0))?);
    println!();

    // Check if parameters have gradients enabled
    println!("Checking requires_grad on policy parameters:");
    let vs = policy.var_store();
    for (name, tensor) in vs.variables() {
        println!("  {}: requires_grad = {}", name, tensor.requires_grad());
    }
    println!();

    // Perform a training step
    println!("Performing training step...");

    let (actions, log_probs, _values) = policy.get_action(&obs);
    println!("  Actions: {:?}", Vec::<i64>::try_from(actions)?);
    println!("  Log probs: {:?}", Vec::<f32>::try_from(&log_probs)?);
    println!();

    // Create fake loss
    let loss = log_probs.mean(Kind::Float).abs();
    println!("  Loss: {:?}", f64::try_from(&loss)?);

    // Backward pass
    optimizer.zero_grad();
    loss.backward();

    // Check if gradients were computed
    println!("\nChecking gradients after backward:");
    for (name, tensor) in vs.variables() {
        let grad = tensor.grad();
        if grad.defined() {
            let grad_norm: f64 = grad.norm().try_into().unwrap_or(0.0);
            if grad_norm > 1e-10 {
                println!("  {}: grad_norm = {:.6}", name, grad_norm);
            } else {
                println!("  {}: grad_norm ≈ 0", name);
            }
        } else {
            println!("  {}: NO GRADIENT (undefined tensor)", name);
        }
    }
    println!();

    // Optimizer step
    optimizer.step();

    // Check if parameters changed
    println!("Checking if parameters changed after optimizer.step():");
    let (logits_after, values_after) = policy.forward(&obs);
    let logits_diff = (&logits_before - &logits_after).abs().mean(Kind::Float);
    let values_diff = (&values_before - &values_after).abs().mean(Kind::Float);

    let logits_diff_val = f64::try_from(&logits_diff)?;
    let values_diff_val = f64::try_from(&values_diff)?;

    println!("  Logits diff: {:.8}", logits_diff_val);
    println!("  Values diff: {:.8}", values_diff_val);

    if logits_diff_val > 1e-6 {
        println!("\n✅ Parameters ARE being updated!");
    } else {
        println!("\n❌ Parameters NOT being updated!");
    }

    Ok(())
}
