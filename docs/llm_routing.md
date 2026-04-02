# Two-Tier LLM Routing

## Design

`LLMRouter` (`app/llm/router.py`) partitions the 18 `SignalType` values into two disjoint tiers using a module-level `frozenset` that is the single source of truth shared by both routing and `DeliberationEngine` escalation:

```python
_FRONTIER_SIGNAL_TYPES: frozenset[SignalType] = frozenset({
    SignalType.CHURN_RISK,
    SignalType.LEGAL_RISK,
    SignalType.SECURITY_CONCERN,
    SignalType.REPUTATION_RISK,
})
```

Any `SignalType` in `_FRONTIER_SIGNAL_TYPES` routes to the primary (frontier) model. All others route to the secondary tier. Both sets are configurable at `LLMServiceConfig` construction time; the `frozenset` itself is the default.

```
request
   │
   ├── signal_type ∈ _FRONTIER_SIGNAL_TYPES ?
   │       YES → primary_model (GPT-4o / Claude 3.5 Sonnet)
   │       NO  → secondary_model (GPT-4o-mini fine-tuned / Ollama)
   │
   └── all routes → Reliability Stack
           retry (exponential back-off, configurable max_retries)
           circuit breaker (CLOSED → OPEN after N failures → HALF_OPEN)
           rate limiting (token bucket per provider)
           timeout management
           cost tracking
```

## Configuration

```python
from app.llm.config import LLMServiceConfig

config = LLMServiceConfig(
    primary_model="gpt-4o",
    fallback_models=["claude-3-5-sonnet-20241022", "gpt-4o-mini"],
    enable_cost_optimization=True,
    max_cost_per_request=0.10,
    max_retries=3,
    retry_initial_delay=1.0,
    retry_max_delay=60.0,
)
```

For fully offline operation (no API keys):

```python
# .env
LOCAL_LLM_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama3.1:8b
```

`LLMRouter` checks `settings.local_llm_url` first. When set, all routing goes through the Ollama REST API at `LOCAL_LLM_URL/api/generate`.

## Routing Strategies

`RoutingStrategy` is an enum passed per-request:

| Strategy | Selection criterion |
|---|---|
| `COST_OPTIMIZED` | Lowest estimated cost per token |
| `QUALITY_OPTIMIZED` | Highest `min_quality_tier` |
| `LATENCY_OPTIMIZED` | Lowest p95 latency from rolling stats |
| `BALANCED` | Weighted combination of cost, quality, latency |
| `ROUND_ROBIN` | Cyclic across eligible providers |
| `AB_TEST` | Traffic split defined by `ABTestConfig.traffic_split` |

```python
from app.llm.router import LLMRouter, RoutingStrategy

router = LLMRouter(service_config=config)

response = await router.generate(
    messages=[LLMMessage(role="user", content="…")],
    strategy=RoutingStrategy.COST_OPTIMIZED,
    temperature=0.2,
    max_tokens=512,
)
```

## Circuit Breaker

State machine per provider: `CLOSED` → `OPEN` (after `cb_open_threshold` consecutive failures) → `HALF_OPEN` (after `cb_half_open_timeout_s` seconds) → `CLOSED` (on next success).

Manual reset:
```python
client = router._get_client("gpt-4o")
await client.circuit_breaker.reset()
```

Circuit breaker state is exposed as Prometheus gauge `llm_circuit_breaker_state` (0=CLOSED, 1=OPEN, 2=HALF_OPEN).

## Prometheus Metrics

| Metric | Type | Labels |
|---|---|---|
| `llm_requests_total` | Counter | `provider`, `model`, `status` |
| `llm_request_duration_seconds` | Histogram | `provider`, `model` |
| `llm_time_to_first_token_seconds` | Histogram | `provider`, `model` |
| `llm_tokens_total` | Counter | `provider`, `model`, `type` (prompt/completion) |
| `llm_cost_total` | Counter | `provider`, `model` |
| `llm_circuit_breaker_state` | Gauge | `provider` |
| `llm_router_decisions` | Counter | `strategy`, `model` |
| `llm_router_fallbacks` | Counter | `primary`, `fallback` |

## LoRA Fine-Tuning (Optional)

Fine-tuning reduces secondary-tier inference cost by ~80% on the 14 non-risk signal types. Training uses QLoRA (4-bit NF4 quantisation).

```python
from app.llm.training import LoRATrainer, LoRATrainingConfig

config = LoRATrainingConfig(
    base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    r=16,               # LoRA rank
    alpha=32,           # scaling factor (2× rank)
    dropout=0.05,
    use_4bit=True,
    bnb_4bit_quant_type="nf4",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler="cosine",
    precision="bf16",
)

trainer = LoRATrainer(config)
trainer.load_model()
metrics = trainer.train(train_dataset, val_dataset)
trainer.save_model("./models/finetuned")
```

Checkpoint structure:
```
checkpoints/<experiment>/
├── checkpoint-epoch1-step500/
│   ├── adapter_model.bin      # LoRA weights only (not full model)
│   ├── adapter_config.json
│   ├── training_state.pt      # optimizer + scheduler + RNG state
│   └── metadata.json
└── final_model/
```

Resume from checkpoint: `--override checkpoint.resume_from_checkpoint=./checkpoints/…`

## Performance Reference (p95, Apple M-series)

| Provider / Model | Latency | Tokens/s |
|---|---|---|
| GPT-4o | 1.2 s | 45 |
| Claude 3.5 Sonnet | 1.8 s | 38 |
| GPT-4o-mini | 0.5 s | 70 |
| Ollama llama3.1:8b (M2 Pro 16 GB) | 3–12 s | 15–40 |

