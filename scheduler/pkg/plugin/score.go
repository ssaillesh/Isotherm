package plugin

import (
	"math"

	pb "isotherm/scheduler/gen/isothermpb"
)

// infeasible marks nodes the simulator says would throttle (or that it doesn't know).
const infeasible int64 = -1

// RawScore turns the simulator's verdict into an orderable integer. It only re-encodes
// the headroom the simulator computed (centi-degC); it does no thermal math of its own.
func RawScore(s *pb.CandidateScore) int64 {
	if s == nil || !s.Feasible || s.HeadroomC < 0 || math.IsNaN(s.HeadroomC) {
		return infeasible
	}
	return int64(math.Round(s.HeadroomC * 100))
}

// Normalize min-max scales feasible raw scores onto [1, maxScore] so the node with the most
// headroom gets maxScore. Infeasible nodes get 0, below every feasible node.
func Normalize(raw []int64, maxScore int64) []int64 {
	out := make([]int64, len(raw))
	lo, hi := int64(math.MaxInt64), int64(math.MinInt64)
	for _, r := range raw {
		if r == infeasible {
			continue
		}
		lo, hi = min(lo, r), max(hi, r)
	}
	for i, r := range raw {
		switch {
		case r == infeasible:
			out[i] = 0
		case hi == lo:
			out[i] = maxScore
		default:
			out[i] = 1 + (r-lo)*(maxScore-1)/(hi-lo)
		}
	}
	return out
}
