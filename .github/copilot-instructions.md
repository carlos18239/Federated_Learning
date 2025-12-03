# Federated Learning System - AI Agent Instructions

## Architecture Overview

This is a **centralized federated learning (FL) platform** implementing the FedAvg algorithm. The system has three main components that communicate via WebSockets:

1. **Agents (Clients)** - `fl_main/agent/client.py`: Hold local data, train models, and send updates to aggregator
2. **Aggregator (Server)** - `fl_main/aggregator/server_th.py`: Coordinates FL rounds, aggregates local models using weighted averaging
3. **PseudoDB** - `fl_main/pseudodb/pseudo_db.py`: Stores model history and metadata in SQLite

## Core Communication Flow

**Registration & Initialization:**
- Agent reads config from `setups/config_agent.json`, generates unique ID via SHA256(MAC address + timestamp)
- Agent sends `participate` message to aggregator with initial model structure (numpy arrays)
- Aggregator responds with `welcome` message containing socket info and current round number
- First agent to connect determines the model shape for the entire FL system

**Training Loop:**
- Agent uses state machine with 4 states tracked via filesystem: `waiting_gm`, `training`, `sending`, `gm_ready`
- State transitions are managed by writing integer values to a state file in `model_path`
- Agent polls or waits for global model arrival, trains locally, sends updates
- Aggregator collects local models until threshold met (`aggregation_threshold` × num_agents), then aggregates via FedAvg
- After aggregation, server pushes cluster models to DB and distributes to all agents

## Critical Implementation Patterns

### Model Serialization
Models are stored as **dictionaries of numpy arrays** with layer names as keys:
```python
{'layer1.weight': np.array(...), 'layer1.bias': np.array(...), ...}
```
Conversion between PyTorch `nn.Module` and this format happens via `Converter` class (see `examples/*/conversion.py`).

### Message Protocol
All messages are **pickle-serialized lists** with type indicator at index 0. Message structure defined by enums in `fl_main/lib/util/states.py`:
- `AgentMsgType`: participate, update, polling
- `AggMsgType`: welcome, update, ack
- Position-based access via `*MSGLocation` enums (e.g., `ParticipateMSGLocation.agent_id`)

### Async Communication
WebSocket handlers use `async/await` with `websockets` library. Key functions:
- `send(msg, ip, socket)` - Send and wait for response
- `receive(websocket)` - Blocking receive
- All servers run on asyncio event loops via `init_fl_server()` or `init_client_server()`

### State Management
`StateManager` (aggregator) tracks volatile state:
- `local_model_buffers`: `LimitedDict` storing collected local models by tensor name
- `cluster_models`: Current aggregated models
- `agent_set`: List of registered agents
- Aggregation trigger: `ready_for_local_aggregation()` checks if enough models collected

## Configuration System

All components read JSON configs from `setups/` directory:
- `config_agent.json`: Aggregator IP, sockets, model paths, polling flag
- `config_aggregator.json`: Sockets, thresholds (`aggregation_threshold`, `round_interval`)
- `config_db.json`: Database paths and connection info

Configs loaded via `set_config_file(component_type)` which expects to be run from workspace root.

## Running the System

**Start components in this order:**
1. Database: `python -m fl_main.pseudodb.pseudo_db`
2. Aggregator: `python -m fl_main.aggregator.server_th`
3. Agents: `python -m examples.heart_disease.binary_clasification`

**Simulation mode** (multiple agents on one machine):
```bash
python -m fl_main.agent.client 1 <exch_socket> <agent_name>
# Args: simulation_flag=1, custom socket, custom name
```

## Example Integration Pattern

Examples (`examples/heart_disease/`, `examples/spaceship-titanic/`) follow this structure:
1. Import `Client` from `fl_main.agent.client`
2. Define `init_models()` returning numpy dict of untrained model
3. Define `training(models_dict)` that converts to PyTorch, trains, converts back
4. Main loop:
   - Send initial models: `client.send_initial_model()`
   - Start client threads: `client.start_fl_client()`
   - Loop: wait for global model → train → send trained model
5. Use `Judge` class for early stopping (max rounds + patience on validation accuracy)

### Model Tracking & Metrics
- Models saved to `data/models/` as `.npz` files with timestamp tags (e.g., `20251113-181153_init.npz`)
- Metrics logged to CSV: `metrics_dataset1.csv` with columns `[timestamp, round, kind, val_acc, test_acc]`
- Final plots generated via `plot_and_save_single_image()` showing local vs global accuracy

## Conventions & Gotchas

**File Paths:** Always use absolute paths. Config expects execution from workspace root (where `setups/` is visible).

**ID Generation:** All IDs are SHA256 hashes of meaningful data (MAC + time for components, component_id + time for models).

**Threading:** Agents run 2 threads: (1) model exchange routine checking state every 5s, (2) websocket server for receiving global models.

**Model Compatibility:** First agent's model structure is authoritative. Subsequent agents must have matching tensor shapes/names.

**Environment:** Uses conda env defined in `setups/federatedenv_linux.yaml`. Key deps: numpy 1.19, websockets 8.1, pytorch (via pip in examples).

## Data Structures

**LimitedDict** (`fl_main/lib/util/data_struc.py`): Custom dict that maintains separate lists per tensor name for aggregation.

**ClientState** enum: 0=waiting_gm, 1=training, 2=sending, 3=gm_ready. Read/write via filesystem.

## Testing & Debugging

- Set `logging.basicConfig(level=logging.DEBUG)` for detailed message inspection
- Check state file in agent's `model_path` directory to debug hanging agents
- Aggregator logs show `ready_for_local_aggregation()` check results each interval
- WebSocket errors often indicate IP mismatch - verify `get_ip()` returns correct address on your network

## Key Files to Reference

- Protocol definitions: `fl_main/lib/util/states.py`
- Message builders: `fl_main/lib/util/messengers.py`
- Helper utilities: `fl_main/lib/util/helpers.py`
- Complete example: `examples/heart_disease/binary_clasification.py`
