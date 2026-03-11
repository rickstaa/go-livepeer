# Realtime Container Autoloader — Design & Implementation Plan

## Status: DRAFT
## Authors: Claude Code + team discussion
## Date: 2025-03-11

---

## 1. Problem Statement

Today, orchestrators can run realtime video pipelines (live-video-to-video) and BYOC
containers, but both paths require **manual container lifecycle management**:

- **Live V2V**: Containers must be pre-warmed via CLI flags or manually started. The
  `livePipelineToImage` map knows what images exist, but there's no on-demand spin-up.
- **BYOC**: Containers must self-register via `POST /capability/register`. The orch has
  zero ability to launch containers itself.

Meanwhile, batch AI pipelines already have `DockerManager.Borrow()` which does on-demand
container creation — but it's not wired into the realtime path.

### Three specific gaps block us:

1. **No GPU type/VRAM awareness** — `allocGPU()` picks the first free GPU slot by string
   ID. It doesn't know if it's an A100 or a 4090, has no VRAM tracking, and can't match
   a model's requirements to hardware. The runner *does* report `GPUComputeInfo` (name,
   memory_total, memory_free, major/minor) via `/hardware`, but this data is collected
   and then **never used for scheduling**.

2. **No fallback or error relay** — When a container crashes mid-stream, the only
   mechanism is the orchestrator swap limiter in `ai_live_video.go` (max 2 swaps per
   3 min). There's no structured error propagation back to the client explaining *why*
   it failed (OOM? image pull failure? GPU incompatible?), no capability downgrade, and
   no hardware-aware retry.

3. **Docker-only runtime** — Current code uses the Docker API directly
   (`DockerManager` → `docker.Client`). This works for single-node prototyping but
   doesn't scale to multi-node, doesn't provide security isolation for untrusted
   containers, and can't leverage k8s scheduling.

---

## 2. How The Industry Solves This

### Container Runtime Progression

| Platform | Runtime | Orchestration | GPU Scheduling | Security |
|----------|---------|---------------|----------------|----------|
| **Chutes** | containerd via K8s | Kubernetes + Gepetto scheduler | K8s device plugin + GraVal GPU verification (95% VRAM matrix test) | cosign image signing, filesystem hashing, Bittensor validator network |
| **Modal** | gVisor (sandbox) + custom runtime | Proprietary scheduler | CUDA-aware placement, multi-GPU grouping | gVisor syscall filtering, snapshot-based isolation, no raw GPU passthrough |
| **RunPod** | Docker + custom orchestrator (NOT k8s) | Proprietary (FlashBoot) | NVIDIA device plugin, multi-GPU (up to 10x24GB) | Secure Cloud = single-tenant machines, Community = container-level isolation |
| **Replicate** | cog (custom OCI) | K8s | K8s GPU scheduling | cog containers are read-only, network-restricted |

### Key Takeaways

1. **Nobody uses raw Docker in production** — they all abstract it behind either K8s or
   a custom scheduler. But all started with Docker for prototyping.

2. **GPU scheduling is always a separate layer** — it's never "pick first free GPU". It's
   always capability-matched (GPU type, VRAM, compute capability).

3. **Security isolation has tiers**:
   - Tier 1 (prototype): Docker + allowlisted registries (where we are)
   - Tier 2 (multi-tenant): kata containers / gVisor + image scanning
   - Tier 3 (production): TEE/confidential computing + signed images + network isolation

4. **Cold start is the killer problem** — everyone optimizes for it differently:
   - Modal: filesystem snapshots + checkpoint/restore
   - RunPod: FlashBoot (<200ms claimed)
   - Chutes: Huggingface model cache on hostPath + image pre-pull

---

## 3. Architecture Decision: Live V2V Pattern (One Capability + ModelID)

As discussed, we'll use the **live-video-to-video capability with model-based routing**
as the foundation, not BYOC's flat capability model. Reasons:

- Existing orchestrator discovery already works with `(capability, modelID)` pairs
- `ModelConstraint` is the natural place to add GPU/VRAM requirements
- Session pool already does model-aware selection
- The `@pipeline` decorator in ai-runner PR #900 aligns perfectly — each decorated
  pipeline becomes a modelID with a known container image

### How ai-runner PR #900 fits

The `@pipeline` decorator creates a standard contract:

```python
@pipeline(name="depth-midas", params=DepthParams)
class DepthMidas:
    @classmethod
    def prepare_models(cls): ...    # build-time model download
    def on_ready(self, **params): ...  # runtime init
    def transform(self, frame, params) -> Tensor: ...  # per-frame
    def on_update(self, **params): ...  # param changes
    def on_stop(self): ...            # cleanup
```

Each `@pipeline(name="X")` maps to:
- A container image: `livepeer/ai-runner:live-app-X`
- A modelID in go-livepeer: `"X"`
- A `ModelConstraint` with GPU/VRAM requirements
- Health check at `/health`, hardware at `/hardware`

This means the autoloader just needs: **modelID → image + GPU requirements → start → register**.

---

## 4. Implementation Plan

### Phase 0: GPU Awareness (PREREQUISITE — ~2 days)

**Goal**: Make GPU allocation hardware-aware so we can match models to GPUs.

#### 4.0.1 GPU Introspection at Startup

```go
// core/gpu.go (new file)
type GPUInfo struct {
    ID            string  // "0", "1", etc.
    Name          string  // "NVIDIA A100-SXM4-80GB"
    VRAMTotal     int64   // bytes
    VRAMFree      int64   // bytes
    ComputeMajor  int
    ComputeMinor  int
}

func ProbeGPUs() ([]GPUInfo, error)
```

Implementation options (in order of preference):
1. Parse `nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader`
2. Use NVML go bindings (`github.com/NVIDIA/go-nvml`)
3. Query the runner's `/hardware` endpoint after first container starts (already works but
   chicken-and-egg for scheduling)

**Decision**: Option 1 for prototype (nvidia-smi is always available on GPU nodes), option
2 for production.

#### 4.0.2 Extend ModelConstraint

```go
// core/capabilities.go
type ModelConstraint struct {
    Warm          bool
    Capacity      int
    RunnerVersion string
    // --- new ---
    MinVRAM        int64    // minimum VRAM in MB, 0 = any
    GPUTypes       []string // allowed GPU types, empty = any
    ContainerImage string   // OCI image ref, empty = use default map
}
```

#### 4.0.3 GPU-Aware allocGPU

```go
func (m *DockerManager) allocGPU(ctx context.Context, req *GPURequirements) (string, error) {
    // 1. Filter GPUs by type if specified
    // 2. Filter by available VRAM
    // 3. Pick best fit (smallest sufficient GPU to avoid waste)
    // 4. Fall back to eviction of non-warm containers on matching GPUs
}
```

### Phase 1: Container Autoloader (CORE — ~3 days)

**Goal**: When a live stream request arrives for modelID X, automatically start the
container if it's not running.

#### 4.1.1 AutoLoader Component

```go
// ai/worker/autoloader.go (new file)
type AutoLoader struct {
    docker      *DockerManager
    gpuInfo     []GPUInfo
    manifests   map[string]*ContainerManifest  // modelID → manifest
    running     map[string]*RunnerContainer     // modelID → container
    mu          sync.Mutex
    idleTimeout time.Duration
}

type ContainerManifest struct {
    ModelID        string
    Pipeline       string
    Image          string
    MinVRAM        int64
    GPUTypes       []string
    Ports          []string
    EnvVars        map[string]string
    HealthEndpoint string  // default: /health
}

// EnsureRunning starts a container for the given modelID if not already running.
// Returns the container's endpoint URL. Blocks until healthy.
func (a *AutoLoader) EnsureRunning(ctx context.Context, modelID string) (string, error)

// Release marks a container as no longer actively serving a stream.
// Idle timeout countdown begins.
func (a *AutoLoader) Release(modelID string)

// EvictIdle runs in background, stops containers idle longer than idleTimeout.
func (a *AutoLoader) EvictIdle(ctx context.Context)
```

#### 4.1.2 Manifest Loading

Manifests loaded from a JSON config file at startup:

```json
{
  "streamdiffusion": {
    "image": "livepeer/ai-runner:live-app-streamdiffusion",
    "pipeline": "live-video-to-video",
    "min_vram_mb": 8000,
    "gpu_types": ["A100", "H100", "RTX4090", "RTX3090"]
  },
  "depth-midas": {
    "image": "livepeer/ai-runner:live-app-depth-midas",
    "pipeline": "live-video-to-video",
    "min_vram_mb": 4000,
    "gpu_types": []
  }
}
```

Future: manifests fetched from a registry service instead of a local file.

#### 4.1.3 Integration Point — stream_orchestrator.go

```go
// byoc/stream_orchestrator.go — StartStream()
func (bso *BYOCOrchestratorServer) StartStream(...) {
    // NEW: auto-load container if needed
    endpoint, err := bso.autoloader.EnsureRunning(ctx, modelID)
    if err != nil {
        // structured error back to gateway
        return err
    }
    // existing flow continues, using endpoint...
}
```

#### 4.1.4 Integration Point — Capability Advertisement

When autoloader starts a container, it auto-registers the capability so orchestrator
discovery works:

```go
func (a *AutoLoader) EnsureRunning(ctx context.Context, modelID string) (string, error) {
    // ... start container ...
    // Auto-register as capability for discovery
    a.orch.RegisterExternalCapability(capSettings)
    return endpoint, nil
}
```

### Phase 2: Fallback & Error Relay (~2 days)

**Goal**: Structured errors and intelligent retry.

#### 4.2.1 Error Classification

```go
// byoc/errors.go (new file)
type ContainerError struct {
    ModelID    string
    OrcAddr    string
    ErrorType  ContainerErrorType
    Message    string
    GPUState   *GPUInfo  // snapshot at time of error
}

type ContainerErrorType int
const (
    ErrImagePullFailed ContainerErrorType = iota
    ErrInsufficientVRAM
    ErrGPUIncompatible
    ErrContainerCrashed
    ErrHealthCheckTimeout
    ErrCapacityExhausted
)
```

#### 4.2.2 Gateway-Side Retry with Error Context

```go
// byoc/stream_gateway.go — enhanced orchestrator selection
func (bsg *BYOCGatewayServer) setupStream(...) {
    var errors []ContainerError
    for _, orch := range orchs {
        endpoint, err := tryOrch(orch, modelID)
        if err != nil {
            errors = append(errors, classifyError(err, orch))
            continue  // try next orch
        }
        return endpoint, nil
    }
    // All orchs failed — return structured error to client
    return nil, &StreamError{Tried: errors, Suggestion: suggestFallback(errors)}
}
```

#### 4.2.3 Error Relay to Client

Errors sent via trickle events channel (already exists):

```json
{
  "type": "error",
  "code": "INSUFFICIENT_VRAM",
  "message": "Model requires 16GB VRAM, orchestrator has 8GB available",
  "tried_orchestrators": 3,
  "suggestion": "Try model streamdiffusion-sd15 (requires 8GB)"
}
```

### Phase 3: Runtime Abstraction (~1 week, future)

**Goal**: Abstract away Docker so we can swap to containerd/K8s.

#### 4.3.1 Container Runtime Interface

```go
// ai/worker/runtime.go (new file)
type ContainerRuntime interface {
    // Create and start a container, return its endpoint
    Start(ctx context.Context, spec ContainerManifest) (endpoint string, err error)
    // Stop and remove a container
    Stop(ctx context.Context, containerID string) error
    // Health check
    Health(ctx context.Context, containerID string) (HealthStatus, error)
    // List running containers
    List(ctx context.Context) ([]ContainerInfo, error)
    // GPU inventory
    GPUs(ctx context.Context) ([]GPUInfo, error)
}

// Implementations:
type DockerRuntime struct { ... }     // Current, wraps DockerManager
type ContainerdRuntime struct { ... } // Future, uses containerd client directly
type K8sRuntime struct { ... }        // Future, creates K8s pods/deployments
```

#### 4.3.2 Why containerd over Docker for next step

- Docker daemon is a single point of failure and adds overhead
- containerd is what Docker uses under the hood anyway
- K8s uses containerd natively — easier migration path
- containerd supports kata containers plugin for security isolation
- RunPod started with Docker, moved to custom containerd-based runtime

#### 4.3.3 Kubernetes Path (later)

For multi-node orchestration:
- K8s with NVIDIA device plugin for GPU scheduling
- Custom scheduler extender for VRAM-aware placement (K8s default GPU scheduling is
  count-based, not VRAM-aware)
- Kata containers for security isolation of untrusted workloads
- Pod security policies to restrict capabilities

### Phase 4: Security for Untrusted Containers (~2 weeks, future)

**Goal**: Let users deploy arbitrary containers safely.

#### 4.4.1 Tiered Security Model

| Tier | Trust Level | Isolation | Registry |
|------|-------------|-----------|----------|
| **Tier 1** (now) | First-party only | Docker container | `livepeer/ai-runner:*` hardcoded |
| **Tier 2** (next) | Allowlisted | Docker + seccomp/AppArmor | Scanned registry with allowlist |
| **Tier 3** (future) | Untrusted | Kata containers (VM-level) | Signed images + vulnerability scan |

#### 4.4.2 Image Security Pipeline

```
User pushes image
    → Registry receives it
    → Trivy/Grype vulnerability scan
    → cosign signature verification (like Chutes)
    → Optional: filesystem hash for reproducibility
    → Image tagged as "verified" in registry metadata
    → Orchestrators only pull "verified" images
```

#### 4.4.3 Kata Containers

Kata runs each container in a lightweight VM — hardware-level isolation:
- Separate kernel per container (no shared kernel exploits)
- GPU passthrough via VFIO (works with NVIDIA GPUs)
- Compatible with containerd and K8s via CRI
- ~50ms overhead per container start (acceptable for our use case)
- **Tradeoff**: More memory overhead per container, GPU passthrough is 1:1 (no MIG/MPS sharing)

#### 4.4.4 Network Isolation

- Containers get no outbound network by default
- Allowlist specific endpoints (model registries, etc.)
- No inter-container communication
- Orchestrator → container communication via localhost only

---

## 5. Implementation Order & Dependencies

```
Phase 0: GPU Awareness
  ├── 0.1 ProbeGPUs (nvidia-smi)
  ├── 0.2 Extend ModelConstraint
  └── 0.3 GPU-aware allocGPU
            │
Phase 1: AutoLoader ◄──────────┘
  ├── 1.1 AutoLoader component
  ├── 1.2 Manifest loading
  ├── 1.3 stream_orchestrator integration
  └── 1.4 Capability auto-registration
            │
Phase 2: Error Relay ◄─────────┘
  ├── 2.1 Error classification
  ├── 2.2 Gateway retry with context
  └── 2.3 Client error relay via trickle
            │
Phase 3: Runtime Abstraction ◄──┘  (can start in parallel)
  ├── 3.1 ContainerRuntime interface
  ├── 3.2 containerd implementation
  └── 3.3 K8s implementation
            │
Phase 4: Security ◄────────────┘
  ├── 4.1 Image scanning pipeline
  ├── 4.2 Kata containers integration
  └── 4.3 Network isolation
```

### Quick Prototype (Phases 0+1): ~5 days

Deliverables:
- `nvidia-smi` GPU probing at orch startup
- `ModelConstraint` extended with VRAM/GPU type/image
- `AutoLoader` that starts containers on-demand for live streams
- Manifest JSON file for container specs
- Auto-registration with existing capability system
- Works with existing ai-runner images AND new `@pipeline` decorator containers

### What this unlocks:
- Orch receives stream request for model "depth-midas"
- AutoLoader checks: is container running? No → start it
- Probes GPUs, finds one with enough VRAM
- Pulls `livepeer/ai-runner:live-app-depth-midas`, starts on matching GPU
- Waits for `/health` to return OK
- Auto-registers capability
- Stream flows through existing trickle infrastructure
- When idle for N minutes, container stopped, GPU freed

---

## 6. Open Questions

1. **Model cache sharing**: Should containers mount a shared model directory (current
   approach via `/models` bind mount) or download models independently? Shared is faster
   for cold start but creates coupling.

2. **Multi-GPU models**: Some models need >1 GPU. Current `allocGPU` returns one GPU.
   Do we need this for the prototype?

3. **Preemption policy**: If all GPUs are busy and a high-priority realtime request
   arrives, should we evict a batch container? What's the priority model?

4. **Manifest distribution**: For production, how do container manifests get distributed?
   On-chain? Via a registry service? P2P?

5. **ai-runner `@pipeline` integration**: Should the decorator auto-generate the manifest
   JSON as part of the image build process? This would close the loop: write Python →
   build image → manifest auto-created → orch auto-loads.
