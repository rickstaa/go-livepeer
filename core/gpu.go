package core

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// GPUInfo represents a physical GPU on the system.
type GPUInfo struct {
	// ID is the GPU index as reported by nvidia-smi (e.g. "0", "1").
	ID string

	// Name is the GPU model name (e.g. "NVIDIA A100-SXM4-80GB").
	Name string

	// VRAMTotalMB is total VRAM in megabytes.
	VRAMTotalMB int64

	// VRAMFreeMB is currently free VRAM in megabytes.
	VRAMFreeMB int64

	// ComputeMajor is the CUDA compute capability major version.
	ComputeMajor int

	// ComputeMinor is the CUDA compute capability minor version.
	ComputeMinor int
}

// MatchesType returns true if this GPU matches any of the given type names.
// Matching is case-insensitive substring match (e.g. "A100" matches "NVIDIA A100-SXM4-80GB").
// An empty types list matches any GPU.
func (g *GPUInfo) MatchesType(types []string) bool {
	if len(types) == 0 {
		return true
	}
	nameLower := strings.ToLower(g.Name)
	for _, t := range types {
		if strings.Contains(nameLower, strings.ToLower(t)) {
			return true
		}
	}
	return false
}

// HasVRAM returns true if the GPU has at least the given VRAM in MB.
// A requirement of 0 matches any GPU.
func (g *GPUInfo) HasVRAM(minMB int64) bool {
	if minMB <= 0 {
		return true
	}
	return g.VRAMTotalMB >= minMB
}

// ProbeGPUs queries nvidia-smi for GPU information.
// Returns an empty slice (not an error) if nvidia-smi is not available,
// which allows graceful fallback on non-GPU machines.
var ProbeGPUs = probeGPUsImpl

func probeGPUsImpl() ([]GPUInfo, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,name,memory.total,memory.free,compute_cap",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		// nvidia-smi not available — not an error, just no GPUs
		return nil, nil
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	gpus := make([]GPUInfo, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		gpu, err := parseNvidiaSmiLine(line)
		if err != nil {
			return nil, fmt.Errorf("parsing nvidia-smi output %q: %w", line, err)
		}
		gpus = append(gpus, gpu)
	}
	return gpus, nil
}

// parseNvidiaSmiLine parses a single CSV line from nvidia-smi.
// Format: index, name, memory.total [MiB], memory.free [MiB], compute_cap
func parseNvidiaSmiLine(line string) (GPUInfo, error) {
	parts := strings.SplitN(line, ", ", 5)
	if len(parts) < 5 {
		return GPUInfo{}, fmt.Errorf("expected 5 fields, got %d", len(parts))
	}

	totalMB, err := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)
	if err != nil {
		return GPUInfo{}, fmt.Errorf("parsing total VRAM: %w", err)
	}
	freeMB, err := strconv.ParseInt(strings.TrimSpace(parts[3]), 10, 64)
	if err != nil {
		return GPUInfo{}, fmt.Errorf("parsing free VRAM: %w", err)
	}

	computeCap := strings.TrimSpace(parts[4])
	major, minor := 0, 0
	if capParts := strings.SplitN(computeCap, ".", 2); len(capParts) == 2 {
		major, _ = strconv.Atoi(capParts[0])
		minor, _ = strconv.Atoi(capParts[1])
	}

	return GPUInfo{
		ID:           strings.TrimSpace(parts[0]),
		Name:         strings.TrimSpace(parts[1]),
		VRAMTotalMB:  totalMB,
		VRAMFreeMB:   freeMB,
		ComputeMajor: major,
		ComputeMinor: minor,
	}, nil
}
