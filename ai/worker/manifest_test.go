package worker

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestManifestRegistry_LoadFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifests.json")

	content := `{
		"streamdiffusion": {
			"pipeline": "live-video-to-video",
			"image": "livepeer/ai-runner:live-app-streamdiffusion",
			"min_vram_mb": 8000,
			"gpu_types": ["A100", "H100"]
		},
		"depth-midas": {
			"pipeline": "live-video-to-video",
			"image": "livepeer/ai-runner:live-app-depth-midas",
			"min_vram_mb": 4000
		}
	}`
	require.NoError(t, os.WriteFile(path, []byte(content), 0644))

	reg := NewManifestRegistry()
	err := reg.LoadFromFile(path)
	require.NoError(t, err)

	// Check streamdiffusion
	m := reg.Get("streamdiffusion")
	require.NotNil(t, m)
	assert.Equal(t, "streamdiffusion", m.ModelID)
	assert.Equal(t, "live-video-to-video", m.Pipeline)
	assert.Equal(t, "livepeer/ai-runner:live-app-streamdiffusion", m.Image)
	assert.Equal(t, int64(8000), m.MinVRAM)
	assert.Equal(t, []string{"A100", "H100"}, m.GPUTypes)

	// Check depth-midas
	m = reg.Get("depth-midas")
	require.NotNil(t, m)
	assert.Equal(t, int64(4000), m.MinVRAM)
	assert.Empty(t, m.GPUTypes) // any GPU

	// Check non-existent
	assert.Nil(t, reg.Get("nonexistent"))
	assert.True(t, reg.Has("streamdiffusion"))
	assert.False(t, reg.Has("nonexistent"))

	// List
	list := reg.List()
	assert.Len(t, list, 2)
}

func TestManifestRegistry_LoadFromLivePipelineMap(t *testing.T) {
	reg := NewManifestRegistry()

	// Pre-load a manifest from file
	reg.Register(&ContainerManifest{
		ModelID:  "streamdiffusion",
		Pipeline: "live-video-to-video",
		Image:    "custom-image:latest",
		MinVRAM:  16000,
	})

	// Load from pipeline map — should NOT overwrite the file-loaded one
	reg.LoadFromLivePipelineMap(map[string]string{
		"streamdiffusion": "livepeer/ai-runner:live-app-streamdiffusion",
		"comfyui":         "livepeer/ai-runner:live-app-comfyui",
	})

	// streamdiffusion should keep the file-loaded version
	m := reg.Get("streamdiffusion")
	require.NotNil(t, m)
	assert.Equal(t, "custom-image:latest", m.Image)
	assert.Equal(t, int64(16000), m.MinVRAM)

	// comfyui should be added from the pipeline map
	m = reg.Get("comfyui")
	require.NotNil(t, m)
	assert.Equal(t, "livepeer/ai-runner:live-app-comfyui", m.Image)
	assert.Equal(t, int64(0), m.MinVRAM) // no VRAM requirement from pipeline map
}

func TestManifestRegistry_LoadFromFile_BadPath(t *testing.T) {
	reg := NewManifestRegistry()
	err := reg.LoadFromFile("/nonexistent/path.json")
	require.Error(t, err)
}

func TestManifestRegistry_LoadFromFile_BadJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	require.NoError(t, os.WriteFile(path, []byte("{bad json"), 0644))

	reg := NewManifestRegistry()
	err := reg.LoadFromFile(path)
	require.Error(t, err)
}
