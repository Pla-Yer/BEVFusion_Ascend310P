# Performance Report

This directory records system performance analysis and optimization results for the BEVFusion-Ascend310P deployment project.

The goal is to document:

- Latency breakdown analysis
- CPU/NPU performance profiling
- Memory usage analysis
- Bottleneck identification
- Optimization strategies and results

Each performance record contains:

- Performance metrics
- Profiling configuration
- Bottleneck analysis
- Optimization results
- Comparison with targets

These records serve as an internal engineering reference and performance knowledge base.

For performance reporting, please use the following template and add an index to Index.md:

## Performance Template

```markdown
## PERF-XXX: [Performance Test Name]

### Date
YYYY-MM-DD

### Configuration
- Hardware:
- Software:
- Model:
- Input Size:

### Metrics
| Component | Latency (ms) | % of Total | Status |
|-----------|--------------|------------|--------|
| Component 1 | | | |

### Bottleneck Analysis
[Identified bottlenecks and root causes]

### Optimization
[Optimization strategies applied]

### Results
[Performance improvement results]

### Notes
[Additional observations]
```
