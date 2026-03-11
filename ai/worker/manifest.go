package worker

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// ContainerManifest describes what a pipeline decorator produces:
// the image to run, the hardware it needs, and how to talk to it.
// This is the go-livepeer side of what @pipeline(name="X") creates in ai-runner.
type ContainerManifest struct {
	// ModelID matches the @pipeline(name="...") in ai-runner.
	ModelID string `json:"model_id"`

	// Pipeline is the Livepeer capability pipeline (e.g. "live-video-to-video").
	Pipeline string `json:"pipeline"`

	// Image is the OCI container image ref (e.g. "livepeer/ai-runner:live-app-depth-midas").
	Image string `json:"image"`

	// MinVRAM is the minimum GPU VRAM required in MB. 0 means any GPU.
	MinVRAM int64 `json:"min_vram_mb"`

	// GPUTypes is an allowlist of compatible GPU names (e.g. ["A100", "H100"]).
	// Empty means any GPU type.
	GPUTypes []string `json:"gpu_types,omitempty"`

	// EnvVars are additional environment variables for the container.
	EnvVars map[string]string `json:"env_vars,omitempty"`

	// IdleTimeout overrides the default idle timeout before the container is evicted.
	// Duration string (e.g. "5m", "30s"). Empty uses the autoloader default.
	IdleTimeout string `json:"idle_timeout,omitempty"`
}

// ManifestRegistry holds the set of known container manifests, keyed by modelID.
// It can be loaded from a JSON file or populated programmatically.
type ManifestRegistry struct {
	mu        sync.RWMutex
	manifests map[string]*ContainerManifest
}

// NewManifestRegistry creates an empty registry.
func NewManifestRegistry() *ManifestRegistry {
	return &ManifestRegistry{
		manifests: make(map[string]*ContainerManifest),
	}
}

// LoadFromFile reads a JSON manifest file and populates the registry.
// The file format is a map of modelID -> ContainerManifest.
//
// Example file:
//
//	{
//	  "streamdiffusion": {
//	    "pipeline": "live-video-to-video",
//	    "image": "livepeer/ai-runner:live-app-streamdiffusion",
//	    "min_vram_mb": 8000,
//	    "gpu_types": ["A100", "H100", "RTX4090"]
//	  },
//	  "depth-midas": {
//	    "pipeline": "live-video-to-video",
//	    "image": "livepeer/ai-runner:live-app-depth-midas",
//	    "min_vram_mb": 4000
//	  }
//	}
func (r *ManifestRegistry) LoadFromFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading manifest file: %w", err)
	}

	var raw map[string]*ContainerManifest
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("parsing manifest file: %w", err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	for modelID, manifest := range raw {
		manifest.ModelID = modelID
		r.manifests[modelID] = manifest
	}
	return nil
}

// LoadFromLivePipelineMap populates the registry from the existing
// livePipelineToImage map, providing backward compatibility with the
// current hardcoded image mapping.
func (r *ManifestRegistry) LoadFromLivePipelineMap(imageMap map[string]string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for modelID, image := range imageMap {
		if _, exists := r.manifests[modelID]; exists {
			// Don't overwrite manifests loaded from file (which have richer metadata)
			continue
		}
		r.manifests[modelID] = &ContainerManifest{
			ModelID:  modelID,
			Pipeline: "live-video-to-video",
			Image:    image,
		}
	}
}

// Register adds or updates a manifest in the registry.
func (r *ManifestRegistry) Register(manifest *ContainerManifest) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.manifests[manifest.ModelID] = manifest
}

// Get returns the manifest for a modelID, or nil if not found.
func (r *ManifestRegistry) Get(modelID string) *ContainerManifest {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.manifests[modelID]
}

// List returns all registered manifests.
func (r *ManifestRegistry) List() []*ContainerManifest {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]*ContainerManifest, 0, len(r.manifests))
	for _, m := range r.manifests {
		result = append(result, m)
	}
	return result
}

// Has returns true if a manifest exists for the given modelID.
func (r *ManifestRegistry) Has(modelID string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.manifests[modelID]
	return ok
}
