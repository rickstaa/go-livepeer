# Autoloader Integration Plan — DockerManager Path

## Status: DRAFT (awaiting approval)
## Date: 2026-03-11
## Prereq: autoloader-prototype-plan.md (original design doc)

---

## Context

The original plan (autoloader-prototype-plan.md) correctly identified the
live-v2v + modelID pattern as the right architecture (Section 3), but Section
4.1.3 incorrectly specified BYOC's `stream_orchestrator.go` as the integration
point. That code was written and reverted.

The correct integration is into **DockerManager** — it already does on-demand
container creation via `Borrow() → createContainer() → allocGPU()`. We enhance
this existing path rather than building a parallel one.

### What already exists (committed but needs rework)

| File | Status | Notes |
|------|--------|-------|
| `core/gpu.go` | **Keep as-is** | `ProbeGPUs()` via nvidia-smi, `GPUInfo` struct with `MatchesType()`/`HasVRAM()` |
| `core/gpu_test.go` | **Keep as-is** | Parsing + matching tests pass |
| `ai/worker/manifest.go` | **Keep as-is** | `ManifestRegistry` with file + legacy map loading |
| `ai/worker/manifest_test.go` | **Keep as-is** | Registry tests pass |
| `ai/worker/autoloader.go` | **Rework** | Was designed for BYOC; needs to work with DockerManager instead |
| `ai/worker/autoloader_test.go` | **Rework** | Tests for utility functions are fine, integration tests need update |

---

## Implementation Steps

### Step 1: GPU-Aware `allocGPU()` in DockerManager

**File**: `ai/worker/docker.go`

**Current** (`allocGPU`): Takes only `ctx`, returns first free GPU string ID.
No VRAM or type awareness.

**Change**: Add a `GPURequirements` parameter and use `core.GPUInfo` data.

```go
// Current signature:
func (m *DockerManager) allocGPU(ctx context.Context) (string, error)

// New signature:
func (m *DockerManager) allocGPU(ctx context.Context, reqs *core.GPURequirements) (string, error)
```

**Logic**:
1. At DockerManager startup, call `core.ProbeGPUs()` and store `[]core.GPUInfo`
2. In `allocGPU()`:
   - If `reqs == nil`, keep current behavior (first free GPU)
   - If `reqs != nil`, filter candidates by `MatchesType()` and `HasVRAM()`
   - Among matching GPUs, prefer smallest sufficient (best-fit)
   - Eviction: only evict containers on GPUs that match requirements
3. Update `createContainer()` call site to pass requirements from `ModelConstraint`

**Callers to update**: `createContainer()` is the only caller of `allocGPU()`.

### Step 2: Extend `ModelConstraint` with GPU Requirements

**File**: `core/capabilities.go`

Add fields to the existing struct:

```go
type ModelConstraint struct {
    Warm          bool
    Capacity      int
    RunnerVersion string
    // New:
    MinVRAMMB      int64    // 0 = any
    GPUTypes       []string // empty = any
    ContainerImage string   // override for getContainerImageName(), empty = use default
}
```

These fields flow from manifest → ModelConstraint → allocGPU requirements.

### Step 3: Manifest-Aware `getContainerImageName()`

**File**: `ai/worker/docker.go`

**Current**: Looks up `livePipelineToImage[modelID]` for live pipelines, with
override support via `m.overrides.Live[modelID]`.

**Change**: Add manifest registry as a third lookup tier:

```
1. m.overrides.Live[modelID]          (user CLI overrides — highest priority)
2. livePipelineToImage[modelID]       (hardcoded map — backward compat)
3. m.manifests.Get(modelID).Image     (manifest file — new pipelines)
4. Error if none found
```

This means new pipelines can be added via manifest JSON without code changes,
while existing hardcoded mappings keep working.

### Step 4: Wire ManifestRegistry into DockerManager

**File**: `ai/worker/docker.go`

- Add `manifests *ManifestRegistry` field to `DockerManager`
- Add `gpuInventory []core.GPUInfo` field to `DockerManager`
- In `NewDockerManager()` or a new init method:
  - Call `core.ProbeGPUs()` → store in `gpuInventory`
  - Create `ManifestRegistry`, load from file if path configured
  - Call `LoadFromLivePipelineMap()` for backward compat
- Pass manifest GPU requirements through to `allocGPU()` in `createContainer()`

### Step 5: Simplify AutoLoader to a Thin DockerManager Wrapper

**File**: `ai/worker/autoloader.go`

The current `AutoLoader` duplicates work that `DockerManager` already does
(container tracking, GPU allocation). Simplify it to only handle:

1. **Idle eviction with timeout** — `DockerManager.watchContainer()` handles
   health but not time-based eviction of non-KeepWarm containers
2. **EnsureRunning()** — Thin wrapper: check if container exists via
   `DockerManager`, if not call `DockerManager.Warm()` to pre-start it
3. **Manifest → ModelConstraint bridging** — Convert manifest GPU requirements
   into ModelConstraint for the capability system

Remove from AutoLoader:
- GPU probing (moved to DockerManager)
- Container tracking (DockerManager already tracks via `m.containers`)
- Direct Docker client usage
- Orchestrator capability registration (keep in DockerManager flow)

### Step 6: Tests

- Unit tests for GPU-aware `allocGPU()` (mock `gpuInventory`)
- Unit tests for manifest fallback in `getContainerImageName()`
- Integration test: manifest-loaded model → Borrow() → correct image + GPU
- Existing `core/gpu_test.go` and `manifest_test.go` remain valid

---

## What This Does NOT Change

- **Batch pipeline path**: Unaffected, batch uses `pipelineToImage` not
  `livePipelineToImage`
- **BYOC path**: Completely untouched (reverted integration stays reverted)
- **Existing hardcoded models**: `livePipelineToImage` keeps working as-is
- **watchContainer() health logic**: No changes to health monitoring
- **Capability advertisement**: Existing flow works, ModelConstraint just gets
  more fields

---

## Order of Work

```
Step 1: GPU-aware allocGPU()
  └─ Depends on: core/gpu.go (already done)
Step 2: Extend ModelConstraint
  └─ Independent
Step 3: Manifest-aware getContainerImageName()
  └─ Depends on: manifest.go (already done)
Step 4: Wire into DockerManager
  └─ Depends on: Steps 1-3
Step 5: Simplify AutoLoader
  └─ Depends on: Step 4
Step 6: Tests
  └─ Alongside each step
```

Steps 1, 2, 3 can be done in parallel. Step 4 integrates them. Step 5 cleans up.

---

## Open Questions

1. **Config flag for manifest file path**: New CLI flag like `--ai-manifest`?
   Or reuse existing config mechanism?
2. **GPU VRAM refresh**: Should we re-probe VRAM periodically (it changes as
   containers allocate), or is startup-only sufficient for prototype?
3. **Eviction timeout default**: What idle duration before stopping a
   non-KeepWarm container? 5 minutes? 10 minutes? Configurable?
