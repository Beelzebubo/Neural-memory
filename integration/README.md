# Neural Memory — Hermes Integration Plugin

Expose neural-memory as a long-term retention tool in [Hermes](https://github.com/anomalyco/hermes) or any registry-compatible agent.

## Plugin Tools

| Tool | Description |
|------|-------------|
| `neural_memory_store` | Store a text memory with metadata. |
| `neural_memory_search` | Semantic search by query text. |
| `neural_memory_recall` | Retrieve a specific memory by ID. |
| `neural_memory_list` | List memories with optional filters. |
| `neural_memory_stats` | Get system statistics. |

## Standalone Usage

```python
from integration.hermes_plugin import MemoryPlugin

plugin = MemoryPlugin()
plugin.cmd_store("Remember this fact", source="chat", importance=0.8)
results = plugin.cmd_search("what was that fact")
print(results)
```

## CLI Tool

```bash
python3 -m integration.cli store "Remember this" --source chat --importance 0.8
python3 -m integration.cli search "what was that"
python3 -m integration.cli list --source chat
python3 -m integration.cli stats
python3 -m integration.cli get <memory-id>
```

## Register with Hermes

### Option 1: Symlink into Hermes plugins directory

```bash
ln -s /path/to/neural-memory/integration /path/to/hermes/plugins/neural_memory
```

Then add to Hermes config:

```json
{
  "plugins": ["neural_memory"]
}
```

### Option 2: Add as a custom tool

In your Hermes config, add:

```json
{
  "custom_tools": [
    {
      "name": "neural_memory",
      "module": "integration.hermes_plugin",
      "class": "MemoryPlugin"
    }
  ]
}
```

### Verify

```bash
hermes tools list
# Should show: neural_memory_store, neural_memory_search, ...
```

## Environment

- `NEURAL_MEMORY_PATH` — path to the store pickle file (default `~/.neural_memory/store.pkl`)
