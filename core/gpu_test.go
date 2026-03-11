package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseNvidiaSmiLine(t *testing.T) {
	tests := []struct {
		name    string
		line    string
		want    GPUInfo
		wantErr bool
	}{
		{
			name: "A100 80GB",
			line: "0, NVIDIA A100-SXM4-80GB, 81920, 81000, 8.0",
			want: GPUInfo{
				ID:           "0",
				Name:         "NVIDIA A100-SXM4-80GB",
				VRAMTotalMB:  81920,
				VRAMFreeMB:   81000,
				ComputeMajor: 8,
				ComputeMinor: 0,
			},
		},
		{
			name: "RTX 4090",
			line: "1, NVIDIA GeForce RTX 4090, 24564, 23000, 8.9",
			want: GPUInfo{
				ID:           "1",
				Name:         "NVIDIA GeForce RTX 4090",
				VRAMTotalMB:  24564,
				VRAMFreeMB:   23000,
				ComputeMajor: 8,
				ComputeMinor: 9,
			},
		},
		{
			name:    "too few fields",
			line:    "0, NVIDIA A100, 81920",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseNvidiaSmiLine(tt.line)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestGPUInfo_MatchesType(t *testing.T) {
	gpu := GPUInfo{Name: "NVIDIA A100-SXM4-80GB"}

	// Empty list matches any
	assert.True(t, gpu.MatchesType(nil))
	assert.True(t, gpu.MatchesType([]string{}))

	// Substring match, case insensitive
	assert.True(t, gpu.MatchesType([]string{"a100"}))
	assert.True(t, gpu.MatchesType([]string{"H100", "A100"}))

	// No match
	assert.False(t, gpu.MatchesType([]string{"RTX4090", "H100"}))
}

func TestGPUInfo_HasVRAM(t *testing.T) {
	gpu := GPUInfo{VRAMTotalMB: 24000}

	assert.True(t, gpu.HasVRAM(0))
	assert.True(t, gpu.HasVRAM(16000))
	assert.True(t, gpu.HasVRAM(24000))
	assert.False(t, gpu.HasVRAM(48000))
}
