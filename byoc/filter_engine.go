package byoc

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Operator is the comparison performed by a Requirement.
type Operator string

const (
	OpIn           Operator = "In"
	OpNotIn        Operator = "NotIn"
	OpExists       Operator = "Exists"
	OpDoesNotExist Operator = "DoesNotExist"
	OpEq           Operator = "Eq"
	OpNotEq        Operator = "NotEq"
	OpGt           Operator = "Gt"
	OpGte          Operator = "Gte"
	OpLt           Operator = "Lt"
	OpLte          Operator = "Lte"
	OpBetween      Operator = "Between"
	OpMatches      Operator = "Matches"
)

// Limits to keep a single filter from being abusive.
const (
	MaxRequirements = 32
	MaxRegexLen     = 256
)

var validOperators = map[Operator]bool{
	OpIn: true, OpNotIn: true, OpExists: true, OpDoesNotExist: true,
	OpEq: true, OpNotEq: true, OpGt: true, OpGte: true, OpLt: true, OpLte: true,
	OpBetween: true, OpMatches: true,
}

// Requirement is one match expression against worker data.
// Key is a dotted path into the worker JSON (e.g. "gpu.memory_mb").
type Requirement struct {
	Key      string   `json:"key"`
	Operator Operator `json:"operator"`
	Values   []string `json:"values,omitempty"`
}

// Selector is a list of Requirements ANDed together.
type Selector struct {
	Version string        `json:"version,omitempty"`
	Match   []Requirement `json:"match,omitempty"`
}

// Sentinel errors. Callers can use errors.Is to distinguish.
var (
	ErrInvalidSelector = errors.New("invalid selector")
	ErrFilterFailed    = errors.New("filter failed")
	ErrKeyMissing      = errors.New("key missing")
	ErrTypeMismatch    = errors.New("type mismatch")
)

// Validate checks operator validity, value cardinality, and compiles regexes.
func (s *Selector) Validate() error {
	if s == nil {
		return nil
	}
	if len(s.Match) > MaxRequirements {
		return fmt.Errorf("%w: too many requirements (%d > %d)", ErrInvalidSelector, len(s.Match), MaxRequirements)
	}
	for i, r := range s.Match {
		if r.Key == "" {
			return fmt.Errorf("%w: requirement %d has empty key", ErrInvalidSelector, i)
		}
		if !validOperators[r.Operator] {
			return fmt.Errorf("%w: requirement %d unknown operator %q", ErrInvalidSelector, i, r.Operator)
		}
		switch r.Operator {
		case OpExists, OpDoesNotExist:
			if len(r.Values) != 0 {
				return fmt.Errorf("%w: %s takes no values", ErrInvalidSelector, r.Operator)
			}
		case OpBetween:
			if len(r.Values) != 2 {
				return fmt.Errorf("%w: Between requires exactly 2 values", ErrInvalidSelector)
			}
			for _, v := range r.Values {
				if _, err := strconv.ParseFloat(v, 64); err != nil {
					return fmt.Errorf("%w: Between bound %q is not numeric", ErrInvalidSelector, v)
				}
			}
		case OpEq, OpNotEq, OpGt, OpGte, OpLt, OpLte:
			if len(r.Values) != 1 {
				return fmt.Errorf("%w: %s requires exactly 1 value", ErrInvalidSelector, r.Operator)
			}
		case OpIn, OpNotIn:
			if len(r.Values) == 0 {
				return fmt.Errorf("%w: %s requires at least 1 value", ErrInvalidSelector, r.Operator)
			}
		case OpMatches:
			if len(r.Values) != 1 {
				return fmt.Errorf("%w: Matches requires exactly 1 value", ErrInvalidSelector)
			}
			if len(r.Values[0]) > MaxRegexLen {
				return fmt.Errorf("%w: regex too long (%d > %d)", ErrInvalidSelector, len(r.Values[0]), MaxRegexLen)
			}
			if _, err := compileAnchored(r.Values[0]); err != nil {
				return fmt.Errorf("%w: bad regex %q: %v", ErrInvalidSelector, r.Values[0], err)
			}
		}
	}
	return nil
}

// Evaluate returns nil if every Requirement is satisfied by data.
// On failure the error wraps ErrFilterFailed/ErrKeyMissing/ErrTypeMismatch with
// the specific Requirement that failed for diagnostics.
func (s *Selector) Evaluate(data map[string]any) error {
	if s == nil || len(s.Match) == 0 {
		return nil
	}
	for _, r := range s.Match {
		if err := evaluateRequirement(r, data); err != nil {
			return err
		}
	}
	return nil
}

func evaluateRequirement(r Requirement, data map[string]any) error {
	val, found := lookupPath(data, r.Key)

	switch r.Operator {
	case OpExists:
		if !found {
			return fmt.Errorf("%w: %s", ErrFilterFailed, r.Key)
		}
		return nil
	case OpDoesNotExist:
		if found {
			return fmt.Errorf("%w: %s exists", ErrFilterFailed, r.Key)
		}
		return nil
	}

	if !found {
		return fmt.Errorf("%w: %s", ErrKeyMissing, r.Key)
	}

	switch r.Operator {
	case OpEq, OpNotEq:
		ok, err := equals(val, r.Values[0])
		if err != nil {
			return err
		}
		if (r.Operator == OpEq) != ok {
			return fmt.Errorf("%w: %s %s %v", ErrFilterFailed, r.Key, r.Operator, r.Values)
		}
		return nil

	case OpGt, OpGte, OpLt, OpLte:
		wf, ok := toFloat(val)
		if !ok {
			return fmt.Errorf("%w: %s expected number, got %T", ErrTypeMismatch, r.Key, val)
		}
		threshold, err := strconv.ParseFloat(r.Values[0], 64)
		if err != nil {
			return fmt.Errorf("%w: %s threshold %q not numeric", ErrInvalidSelector, r.Key, r.Values[0])
		}
		var pass bool
		switch r.Operator {
		case OpGt:
			pass = wf > threshold
		case OpGte:
			pass = wf >= threshold
		case OpLt:
			pass = wf < threshold
		case OpLte:
			pass = wf <= threshold
		}
		if !pass {
			return fmt.Errorf("%w: %s %s %s (got %v)", ErrFilterFailed, r.Key, r.Operator, r.Values[0], wf)
		}
		return nil

	case OpBetween:
		wf, ok := toFloat(val)
		if !ok {
			return fmt.Errorf("%w: %s expected number, got %T", ErrTypeMismatch, r.Key, val)
		}
		minV, _ := strconv.ParseFloat(r.Values[0], 64)
		maxV, _ := strconv.ParseFloat(r.Values[1], 64)
		if wf < minV || wf > maxV {
			return fmt.Errorf("%w: %s not in [%v,%v] (got %v)", ErrFilterFailed, r.Key, minV, maxV, wf)
		}
		return nil

	case OpIn, OpNotIn:
		hit, err := membership(val, r.Values)
		if err != nil {
			return err
		}
		if (r.Operator == OpIn) != hit {
			return fmt.Errorf("%w: %s %s %v", ErrFilterFailed, r.Key, r.Operator, r.Values)
		}
		return nil

	case OpMatches:
		s, ok := val.(string)
		if !ok {
			return fmt.Errorf("%w: %s expected string for Matches, got %T", ErrTypeMismatch, r.Key, val)
		}
		re, err := compileAnchored(r.Values[0])
		if err != nil {
			return fmt.Errorf("%w: %s bad regex: %v", ErrInvalidSelector, r.Key, err)
		}
		if !re.MatchString(s) {
			return fmt.Errorf("%w: %s !~ %s", ErrFilterFailed, r.Key, r.Values[0])
		}
		return nil
	}

	return fmt.Errorf("%w: unhandled operator %s", ErrInvalidSelector, r.Operator)
}

// lookupPath walks dotted-path through nested map[string]any.
// Stops at the first non-map value, returning whatever it finds at the leaf.
func lookupPath(data map[string]any, path string) (any, bool) {
	if data == nil {
		return nil, false
	}
	parts := strings.Split(path, ".")
	var cur any = data
	for i, part := range parts {
		m, ok := cur.(map[string]any)
		if !ok {
			return nil, false
		}
		v, ok := m[part]
		if !ok {
			return nil, false
		}
		if i == len(parts)-1 {
			return v, true
		}
		cur = v
	}
	return nil, false
}

// toFloat normalizes any numeric JSON shape to float64.
// Strings are intentionally not coerced — type mismatch is a real signal.
func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		if err != nil {
			return 0, false
		}
		return f, true
	}
	return 0, false
}

// equals compares a worker value against a string literal from the filter.
// Numbers compare numerically (so filter "8000" matches worker 8000.0).
// Strings compare exactly. Other types return type mismatch.
func equals(workerVal any, filterVal string) (bool, error) {
	switch w := workerVal.(type) {
	case string:
		return w == filterVal, nil
	case bool:
		fb, err := strconv.ParseBool(filterVal)
		if err != nil {
			return false, fmt.Errorf("%w: expected bool, got literal %q", ErrTypeMismatch, filterVal)
		}
		return w == fb, nil
	}
	if wf, ok := toFloat(workerVal); ok {
		ff, err := strconv.ParseFloat(filterVal, 64)
		if err != nil {
			return false, fmt.Errorf("%w: expected number, got literal %q", ErrTypeMismatch, filterVal)
		}
		return wf == ff, nil
	}
	return false, fmt.Errorf("%w: cannot compare %T to literal", ErrTypeMismatch, workerVal)
}

// membership tests whether workerVal is "in" the filter values list.
// Scalar worker value: equality against any filter value.
// Array worker value: any element equal to any filter value.
func membership(workerVal any, filterVals []string) (bool, error) {
	if arr, ok := workerVal.([]any); ok {
		for _, elem := range arr {
			for _, fv := range filterVals {
				ok, err := equals(elem, fv)
				if err != nil {
					continue // skip incomparable elements rather than failing the whole check
				}
				if ok {
					return true, nil
				}
			}
		}
		return false, nil
	}
	for _, fv := range filterVals {
		ok, err := equals(workerVal, fv)
		if err != nil {
			return false, err
		}
		if ok {
			return true, nil
		}
	}
	return false, nil
}

// compileAnchored wraps the pattern with ^...$ unless already anchored.
// Documented behavior: Matches is full-string anchored.
func compileAnchored(pattern string) (*regexp.Regexp, error) {
	p := pattern
	if !strings.HasPrefix(p, "^") {
		p = "^" + p
	}
	if !strings.HasSuffix(p, "$") {
		p = p + "$"
	}
	return regexp.Compile(p)
}
