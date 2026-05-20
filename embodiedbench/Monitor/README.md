# WP4 Safety Brain Monitor

Runtime safety monitoring module for the BIEA robot system.

## Usage

```python
from embodiedbench.Monitor import SafetyBrainMonitor

monitor = SafetyBrainMonitor(model="qwen-plus")
# Implements MonitorInterface.check(observation, memory_snapshot)
```

## Requires

- `DASHSCOPE_API_KEY` environment variable (for qwen-plus LLM)
- `requests` package
