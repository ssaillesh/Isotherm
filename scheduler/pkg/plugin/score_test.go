package plugin

import (
	"math"
	"reflect"
	"testing"

	pb "isotherm/scheduler/gen/isothermpb"
)

func TestRawScore(t *testing.T) {
	cases := []struct {
		name string
		in   *pb.CandidateScore
		want int64
	}{
		{"nil", nil, infeasible},
		{"infeasible", &pb.CandidateScore{Feasible: false, HeadroomC: 10}, infeasible},
		{"negative headroom", &pb.CandidateScore{Feasible: true, HeadroomC: -0.1}, infeasible},
		{"nan", &pb.CandidateScore{Feasible: true, HeadroomC: math.NaN()}, infeasible},
		{"feasible", &pb.CandidateScore{Feasible: true, HeadroomC: 12.345}, 1235},
	}
	for _, c := range cases {
		if got := RawScore(c.in); got != c.want {
			t.Errorf("%s: RawScore = %d, want %d", c.name, got, c.want)
		}
	}
}

func TestNormalize(t *testing.T) {
	cases := []struct {
		name string
		raw  []int64
		want []int64
	}{
		{"empty", nil, []int64{}},
		{"spread", []int64{100, 600, 1100}, []int64{1, 50, 100}},
		{"infeasible ranks last", []int64{infeasible, 200, 400}, []int64{0, 1, 100}},
		{"all equal", []int64{500, 500}, []int64{100, 100}},
		{"all infeasible", []int64{infeasible, infeasible}, []int64{0, 0}},
	}
	for _, c := range cases {
		if got := Normalize(c.raw, 100); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: Normalize(%v) = %v, want %v", c.name, c.raw, got, c.want)
		}
	}
}
