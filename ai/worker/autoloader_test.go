package worker

import (
	"fmt"
	"testing"

	"github.com/livepeer/go-livepeer/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFindCompatibleGPU(t *testing.T) {
	gpus := []core.GPUInfo{
		{ID: "0", Name: "NVIDIA A100-SXM4-80GB", VRAMTotalMB: 81920},
		{ID: "1", Name: "NVIDIA GeForce RTX 4090", VRAMTotalMB: 24564},
		{ID: "2", Name: "NVIDIA GeForce RTX 3090", VRAMTotalMB: 24576},
	}

	tests := []struct {
		name   string
		req    GPURequirements
		wantID string
		found  bool
	}{
		{
			name:   "any GPU, no VRAM requirement",
			req:    GPURequirements{},
			wantID: "1", // smallest (best fit)
			found:  true,
		},
		{
			name:   "needs 48GB VRAM",
			req:    GPURequirements{MinVRAMMB: 48000},
			wantID: "0", // only A100 has enough
			found:  true,
		},
		{
			name:   "needs 200GB — nothing fits",
			req:    GPURequirements{MinVRAMMB: 200000},
			wantID: "",
			found:  false,
		},
		{
			name:   "A100 type filter",
			req:    GPURequirements{GPUTypes: []string{"A100"}},
			wantID: "0",
			found:  true,
		},
		{
			name:   "RTX type filter with VRAM",
			req:    GPURequirements{GPUTypes: []string{"RTX"}, MinVRAMMB: 24000},
			wantID: "1", // smallest RTX that fits
			found:  true,
		},
		{
			name:   "H100 not available",
			req:    GPURequirements{GPUTypes: []string{"H100"}},
			wantID: "",
			found:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gpu := FindCompatibleGPU(gpus, tt.req)
			if !tt.found {
				assert.Nil(t, gpu)
				return
			}
			require.NotNil(t, gpu)
			assert.Equal(t, tt.wantID, gpu.ID)
		})
	}
}

func TestClassifyContainerError(t *testing.T) {
	tests := []struct {
		errMsg   string
		wantType ContainerErrorType
	}{
		{"insufficient capacity", ErrTypeCapacityExhausted},
		{"no compatible GPU available", ErrTypeGPUIncompatible},
		{"needs 16GB VRAM", ErrTypeInsufficientVRAM},
		{"failed to pull image", ErrTypeImagePull},
		{"timed out waiting for runner", ErrTypeHealthTimeout},
		{"container exited unexpectedly", ErrTypeContainerCrash},
	}

	for _, tt := range tests {
		t.Run(tt.errMsg, func(t *testing.T) {
			ce := ClassifyContainerError("test-model", fmt.Errorf(tt.errMsg))
			assert.Equal(t, tt.wantType, ce.Type)
			assert.Equal(t, "test-model", ce.ModelID)
			assert.Contains(t, ce.Error(), tt.errMsg)
		})
	}
}

func TestManifestToGPURequirements(t *testing.T) {
	m := &ContainerManifest{
		MinVRAM:  8000,
		GPUTypes: []string{"A100", "H100"},
	}
	req := ManifestToGPURequirements(m)
	assert.Equal(t, int64(8000), req.MinVRAMMB)
	assert.Equal(t, []string{"A100", "H100"}, req.GPUTypes)
}
