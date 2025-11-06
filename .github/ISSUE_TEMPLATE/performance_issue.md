---
name: Performance Issue
about: Report slow performance or optimization opportunity
title: '[PERF] '
labels: performance
assignees: ''
---

## ⚡ Performance Issue

Describe the performance problem you're experiencing.

## 📊 Benchmarks

Please provide benchmark results if available:

```bash
# Command used
cargo bench -- my_benchmark

# Results
test my_benchmark ... bench: 1,234,567 ns/iter (+/- 123,456)
```

## 🔍 Profiling Data

If you've profiled the code, include relevant information:
- Flamegraph screenshots
- CPU/memory usage
- GPU utilization
- Bottleneck identification

## 💻 Environment

- **OS:** [e.g., Ubuntu 22.04]
- **CPU:** [e.g., AMD Ryzen 9 5950X]
- **RAM:** [e.g., 32GB DDR4]
- **GPU:** [e.g., NVIDIA RTX 4090]
- **Thrust Version:** [e.g., 0.1.0]

## 📝 Expected Performance

What performance did you expect? (e.g., "Should achieve 1M SPS like PufferLib")

## 🎯 Actual Performance

What performance are you seeing? (e.g., "Only achieving 200K SPS")

## 🔧 Proposed Solution

(Optional) Ideas for optimization.

## 📈 Success Criteria

How will we know when this is fixed?
- [ ] Achieves X SPS
- [ ] Reduces memory usage by Y%
- [ ] Matches benchmark Z
