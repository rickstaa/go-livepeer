package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/livepeer/go-livepeer/core"
)

var (
	defaultIdleTimeout   = 5 * time.Minute
	defaultStartTimeout  = 4 * time.Minute
	idleCheckInterval    = 30 * time.Second
	ErrNoManifest        = fmt.Errorf("no manifest found for model")
	ErrNoCompatibleGPU   = fmt.Errorf("no compatible GPU available")
	ErrContainerStarting = fmt.Errorf("container is starting")
)

// GPURequirements specifies what hardware a container needs.
type GPURequirements struct {
	MinVRAMMB int64
	GPUTypes  []string
}

// AutoLoaderOrchestrator is the subset of orchestrator functions
// the autoloader needs to register/unregister capabilities.
type AutoLoaderOrchestrator interface {
	RegisterExternalCapability(extCapSettings string) (*core.ExternalCapability, error)
	RemoveExternalCapability(extCapName string) error
}

// containerState tracks a running container managed by the autoloader.
type containerState struct {
	modelID     string
	container   *RunnerContainer
	endpoint    string
	lastUsed    time.Time
	activeCount int // number of active streams using this container
	mu          sync.Mutex
}

func (cs *containerState) acquire() {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.activeCount++
	cs.lastUsed = time.Now()
}

func (cs *containerState) release() {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	if cs.activeCount > 0 {
		cs.activeCount--
	}
	cs.lastUsed = time.Now()
}

func (cs *containerState) isIdle() bool {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	return cs.activeCount == 0
}

// AutoLoader manages on-demand container lifecycle for realtime pipelines.
// It starts containers when streams request a modelID, keeps them running
// while active, and evicts them after an idle timeout.
type AutoLoader struct {
	docker    *DockerManager
	gpuInfo   []core.GPUInfo
	manifests *ManifestRegistry
	orch      AutoLoaderOrchestrator

	mu       sync.Mutex
	running  map[string]*containerState // modelID -> state
	starting map[string]chan struct{}    // modelID -> channel closed when ready

	idleTimeout time.Duration
	ctx         context.Context
	cancel      context.CancelFunc
}

// AutoLoaderConfig holds configuration for the AutoLoader.
type AutoLoaderConfig struct {
	Docker      *DockerManager
	Manifests   *ManifestRegistry
	Orch        AutoLoaderOrchestrator
	IdleTimeout time.Duration
}

// NewAutoLoader creates an AutoLoader and probes GPUs at startup.
func NewAutoLoader(cfg AutoLoaderConfig) (*AutoLoader, error) {
	gpus, err := core.ProbeGPUs()
	if err != nil {
		return nil, fmt.Errorf("probing GPUs: %w", err)
	}

	if len(gpus) > 0 {
		slog.Info("AutoLoader GPU inventory",
			slog.Int("count", len(gpus)))
		for _, g := range gpus {
			slog.Info("  GPU",
				slog.String("id", g.ID),
				slog.String("name", g.Name),
				slog.Int64("vram_total_mb", g.VRAMTotalMB),
				slog.Int64("vram_free_mb", g.VRAMFreeMB))
		}
	} else {
		slog.Warn("AutoLoader: no GPUs detected, hardware-aware scheduling disabled")
	}

	idleTimeout := cfg.IdleTimeout
	if idleTimeout == 0 {
		idleTimeout = defaultIdleTimeout
	}

	ctx, cancel := context.WithCancel(context.Background())
	al := &AutoLoader{
		docker:      cfg.Docker,
		gpuInfo:     gpus,
		manifests:   cfg.Manifests,
		orch:        cfg.Orch,
		running:     make(map[string]*containerState),
		starting:    make(map[string]chan struct{}),
		idleTimeout: idleTimeout,
		ctx:         ctx,
		cancel:      cancel,
	}

	go al.evictIdleLoop()

	return al, nil
}

// EnsureRunning guarantees a container is running for the given modelID.
// If already running, returns immediately. If starting, waits for it.
// If not started, starts it and blocks until healthy.
// Returns the container endpoint URL.
func (a *AutoLoader) EnsureRunning(ctx context.Context, modelID string) (string, error) {
	// Fast path: already running
	a.mu.Lock()
	if cs, ok := a.running[modelID]; ok {
		cs.acquire()
		a.mu.Unlock()
		return cs.endpoint, nil
	}

	// Check if another goroutine is already starting this container
	if ch, ok := a.starting[modelID]; ok {
		a.mu.Unlock()
		select {
		case <-ch:
			// Started, grab the running state
			a.mu.Lock()
			cs, ok := a.running[modelID]
			a.mu.Unlock()
			if !ok {
				return "", fmt.Errorf("container for %s failed to start", modelID)
			}
			cs.acquire()
			return cs.endpoint, nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}

	// We're the first — claim the start
	startCh := make(chan struct{})
	a.starting[modelID] = startCh
	a.mu.Unlock()

	endpoint, err := a.startContainer(ctx, modelID)

	a.mu.Lock()
	delete(a.starting, modelID)
	close(startCh)
	if err != nil {
		a.mu.Unlock()
		return "", err
	}
	cs := a.running[modelID]
	cs.acquire()
	a.mu.Unlock()
	return endpoint, nil
}

// Release marks a stream as no longer using a container.
// The container will be evicted after the idle timeout if no other streams use it.
func (a *AutoLoader) Release(modelID string) {
	a.mu.Lock()
	cs, ok := a.running[modelID]
	a.mu.Unlock()
	if ok {
		cs.release()
	}
}

// Stop shuts down the autoloader and all managed containers.
func (a *AutoLoader) Stop() {
	a.cancel()
}

// startContainer does the actual work of starting a container for a modelID.
func (a *AutoLoader) startContainer(ctx context.Context, modelID string) (string, error) {
	manifest := a.manifests.Get(modelID)
	if manifest == nil {
		return "", fmt.Errorf("%w: %s", ErrNoManifest, modelID)
	}

	// Check GPU compatibility if we have GPU info
	if len(a.gpuInfo) > 0 {
		compatible := false
		for _, gpu := range a.gpuInfo {
			if gpu.MatchesType(manifest.GPUTypes) && gpu.HasVRAM(manifest.MinVRAM) {
				compatible = true
				break
			}
		}
		if !compatible {
			return "", fmt.Errorf("%w: model %s needs %dMB VRAM, types %v",
				ErrNoCompatibleGPU, modelID, manifest.MinVRAM, manifest.GPUTypes)
		}
	}

	slog.Info("AutoLoader starting container",
		slog.String("model_id", modelID),
		slog.String("image", manifest.Image),
		slog.Int64("min_vram_mb", manifest.MinVRAM))

	startCtx, cancel := context.WithTimeout(ctx, defaultStartTimeout)
	defer cancel()

	// Use the existing DockerManager.Warm() path which handles:
	// - GPU allocation (allocGPU)
	// - Image pull if needed
	// - Container creation with GPU passthrough
	// - Health check wait
	// - Watch goroutine
	err := a.docker.Warm(startCtx, manifest.Pipeline, modelID, nil)
	if err != nil {
		return "", fmt.Errorf("starting container for %s: %w", modelID, err)
	}

	// Find the container that was just created to get its endpoint
	endpoint := a.findContainerEndpoint(modelID)
	if endpoint == "" {
		return "", fmt.Errorf("container started but endpoint not found for %s", modelID)
	}

	// Register as external capability so orchestrator discovery works
	if a.orch != nil {
		a.registerCapability(modelID, manifest, endpoint)
	}

	a.mu.Lock()
	a.running[modelID] = &containerState{
		modelID:  modelID,
		endpoint: endpoint,
		lastUsed: time.Now(),
	}
	a.mu.Unlock()

	slog.Info("AutoLoader container ready",
		slog.String("model_id", modelID),
		slog.String("endpoint", endpoint))

	return endpoint, nil
}

// findContainerEndpoint looks through the DockerManager's containers to find
// the endpoint for a given modelID.
func (a *AutoLoader) findContainerEndpoint(modelID string) string {
	a.docker.mu.Lock()
	defer a.docker.mu.Unlock()
	for _, rc := range a.docker.containers {
		if rc.ModelID == modelID {
			return rc.Endpoint.URL
		}
	}
	return ""
}

// registerCapability registers a running container as an external capability
// on the orchestrator so it can be discovered.
func (a *AutoLoader) registerCapability(modelID string, manifest *ContainerManifest, endpoint string) {
	capJSON, err := json.Marshal(map[string]interface{}{
		"name":           modelID,
		"url":            endpoint,
		"capacity":       1,
		"price_per_unit": 0,
		"price_scaling":  1,
	})
	if err != nil {
		slog.Error("AutoLoader failed to marshal capability",
			slog.String("model_id", modelID),
			slog.String("error", err.Error()))
		return
	}

	_, err = a.orch.RegisterExternalCapability(string(capJSON))
	if err != nil {
		slog.Error("AutoLoader failed to register capability",
			slog.String("model_id", modelID),
			slog.String("error", err.Error()))
		return
	}

	slog.Info("AutoLoader registered capability",
		slog.String("model_id", modelID),
		slog.String("endpoint", endpoint))
}

// evictIdleLoop periodically checks for idle containers and stops them.
func (a *AutoLoader) evictIdleLoop() {
	ticker := time.NewTicker(idleCheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			a.evictIdle()
		}
	}
}

func (a *AutoLoader) evictIdle() {
	a.mu.Lock()
	var toEvict []string
	now := time.Now()
	for modelID, cs := range a.running {
		if cs.isIdle() {
			cs.mu.Lock()
			idle := now.Sub(cs.lastUsed)
			cs.mu.Unlock()
			if idle >= a.idleTimeout {
				toEvict = append(toEvict, modelID)
			}
		}
	}
	a.mu.Unlock()

	for _, modelID := range toEvict {
		a.stopContainer(modelID)
	}
}

func (a *AutoLoader) stopContainer(modelID string) {
	a.mu.Lock()
	cs, ok := a.running[modelID]
	if !ok {
		a.mu.Unlock()
		return
	}
	delete(a.running, modelID)
	a.mu.Unlock()

	slog.Info("AutoLoader evicting idle container",
		slog.String("model_id", modelID))

	// Unregister capability
	if a.orch != nil {
		if err := a.orch.RemoveExternalCapability(modelID); err != nil {
			slog.Error("AutoLoader failed to unregister capability",
				slog.String("model_id", modelID),
				slog.String("error", err.Error()))
		}
	}

	// Stop the container via DockerManager
	a.docker.mu.Lock()
	for _, rc := range a.docker.containers {
		if rc.ModelID == modelID {
			a.docker.mu.Unlock()
			if err := a.docker.destroyContainer(rc, false); err != nil {
				slog.Error("AutoLoader failed to destroy container",
					slog.String("model_id", modelID),
					slog.String("error", err.Error()))
			}
			return
		}
	}
	a.docker.mu.Unlock()

	_ = cs // suppress unused warning
}

// IsRunning returns true if a container is currently running for the given modelID.
func (a *AutoLoader) IsRunning(modelID string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	_, ok := a.running[modelID]
	return ok
}

// RunningModels returns the list of currently running modelIDs.
func (a *AutoLoader) RunningModels() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	models := make([]string, 0, len(a.running))
	for m := range a.running {
		models = append(models, m)
	}
	return models
}

// GPUInventory returns the detected GPUs.
func (a *AutoLoader) GPUInventory() []core.GPUInfo {
	return a.gpuInfo
}

// FindCompatibleGPU returns the first GPU that meets the requirements.
// Returns nil if no compatible GPU is found.
func FindCompatibleGPU(gpus []core.GPUInfo, req GPURequirements) *core.GPUInfo {
	// Try to find smallest sufficient GPU (best fit) to avoid waste
	var best *core.GPUInfo
	for i := range gpus {
		gpu := &gpus[i]
		if !gpu.MatchesType(req.GPUTypes) {
			continue
		}
		if !gpu.HasVRAM(req.MinVRAMMB) {
			continue
		}
		if best == nil || gpu.VRAMTotalMB < best.VRAMTotalMB {
			best = gpu
		}
	}
	return best
}

// ManifestToGPURequirements converts a ContainerManifest to GPURequirements.
func ManifestToGPURequirements(m *ContainerManifest) GPURequirements {
	return GPURequirements{
		MinVRAMMB: m.MinVRAM,
		GPUTypes:  m.GPUTypes,
	}
}

// ContainerErrorType classifies container failures for structured error relay.
type ContainerErrorType int

const (
	ErrTypeImagePull ContainerErrorType = iota
	ErrTypeInsufficientVRAM
	ErrTypeGPUIncompatible
	ErrTypeContainerCrash
	ErrTypeHealthTimeout
	ErrTypeCapacityExhausted
)

var containerErrorTypeNames = map[ContainerErrorType]string{
	ErrTypeImagePull:         "IMAGE_PULL_FAILED",
	ErrTypeInsufficientVRAM:  "INSUFFICIENT_VRAM",
	ErrTypeGPUIncompatible:   "GPU_INCOMPATIBLE",
	ErrTypeContainerCrash:    "CONTAINER_CRASHED",
	ErrTypeHealthTimeout:     "HEALTH_CHECK_TIMEOUT",
	ErrTypeCapacityExhausted: "CAPACITY_EXHAUSTED",
}

func (t ContainerErrorType) String() string {
	if name, ok := containerErrorTypeNames[t]; ok {
		return name
	}
	return "UNKNOWN"
}

// ContainerError is a structured error for container failures.
type ContainerError struct {
	ModelID   string             `json:"model_id"`
	Type      ContainerErrorType `json:"error_type"`
	TypeName  string             `json:"error_code"`
	Message   string             `json:"message"`
	GPUState  *core.GPUInfo      `json:"gpu_state,omitempty"`
	OrcAddr   string             `json:"orchestrator,omitempty"`
}

func (e *ContainerError) Error() string {
	return fmt.Sprintf("[%s] %s: %s", e.TypeName, e.ModelID, e.Message)
}

// ClassifyContainerError inspects an error and returns a structured ContainerError.
func ClassifyContainerError(modelID string, err error) *ContainerError {
	msg := err.Error()
	errType := ErrTypeContainerCrash

	switch {
	case strings.Contains(msg, "insufficient capacity"):
		errType = ErrTypeCapacityExhausted
	case strings.Contains(msg, "no compatible GPU"):
		errType = ErrTypeGPUIncompatible
	case strings.Contains(msg, "VRAM"):
		errType = ErrTypeInsufficientVRAM
	case strings.Contains(msg, "pull"):
		errType = ErrTypeImagePull
	case strings.Contains(msg, "timed out") || strings.Contains(msg, "health"):
		errType = ErrTypeHealthTimeout
	}

	return &ContainerError{
		ModelID:  modelID,
		Type:     errType,
		TypeName: errType.String(),
		Message:  msg,
	}
}
