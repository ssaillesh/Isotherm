package plugin

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/grpc"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"

	pb "isotherm/scheduler/gen/isothermpb"
)

// fakeSim answers ScoreCandidates with canned headroom and records binds.
type fakeSim struct {
	pb.ThermalSimClient
	headroom map[string]float64
	err      error
	lastReq  *pb.ScoreRequest
	binds    []*pb.PodEvent
}

func (f *fakeSim) ScoreCandidates(_ context.Context, in *pb.ScoreRequest, _ ...grpc.CallOption) (*pb.ScoreResponse, error) {
	f.lastReq = in
	if f.err != nil {
		return nil, f.err
	}
	resp := &pb.ScoreResponse{}
	for _, n := range in.CandidateNodes {
		h, ok := f.headroom[n]
		resp.Scores = append(resp.Scores, &pb.CandidateScore{Node: n, HeadroomC: h, Feasible: ok})
	}
	return resp, nil
}

func (f *fakeSim) ApplyEvent(_ context.Context, in *pb.PodEvent, _ ...grpc.CallOption) (*pb.ApplyEventResponse, error) {
	f.binds = append(f.binds, in)
	return &pb.ApplyEventResponse{Ok: true}, nil
}

func gpuPod(gpus int64) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:   "default",
			Name:        "job-1",
			Annotations: map[string]string{AnnotationGPUUtil: "0.9", AnnotationDuration: "300"},
		},
		Spec: v1.PodSpec{Containers: []v1.Container{{
			Resources: v1.ResourceRequirements{Limits: v1.ResourceList{
				GPUResource: *resource.NewQuantity(gpus, resource.DecimalSI),
			}},
		}}},
	}
}

func nodeInfos(names ...string) []*framework.NodeInfo {
	out := make([]*framework.NodeInfo, 0, len(names))
	for _, n := range names {
		ni := framework.NewNodeInfo()
		ni.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: n}})
		out = append(out, ni)
	}
	return out
}

func TestWorkloadFromPod(t *testing.T) {
	w := WorkloadFromPod(gpuPod(4))
	if w.WorkloadId != "default/job-1" || w.GpuCount != 4 || w.GpuUtil != 0.9 || w.DurationS != 300 {
		t.Fatalf("unexpected workload: %v", w)
	}
}

func TestScoresFollowSimulatorHeadroom(t *testing.T) {
	ctx := context.Background()
	sim := &fakeSim{headroom: map[string]float64{"cool": 12, "warm": 3}}
	p := NewWithClient(sim)
	state := framework.NewCycleState()
	pod := gpuPod(2)

	if st := p.PreScore(ctx, state, pod, nodeInfos("cool", "warm", "hot")); !st.IsSuccess() {
		t.Fatalf("PreScore: %v", st)
	}
	if got := len(sim.lastReq.CandidateNodes); got != 3 {
		t.Fatalf("expected one batched RPC with 3 candidates, got %d", got)
	}

	var scores framework.NodeScoreList
	for _, n := range []string{"cool", "warm", "hot"} {
		s, st := p.Score(ctx, state, pod, n)
		if !st.IsSuccess() {
			t.Fatalf("Score(%s): %v", n, st)
		}
		scores = append(scores, framework.NodeScore{Name: n, Score: s})
	}
	if st := p.NormalizeScore(ctx, state, pod, scores); !st.IsSuccess() {
		t.Fatalf("NormalizeScore: %v", st)
	}
	if !(scores[0].Score == framework.MaxNodeScore && scores[1].Score > 0 && scores[2].Score == 0) {
		t.Fatalf("want cool > warm > hot(infeasible), got %v", scores)
	}
}

func TestPreScoreSkipsWhenSimDown(t *testing.T) {
	p := NewWithClient(&fakeSim{err: errors.New("unavailable")})
	st := p.PreScore(context.Background(), framework.NewCycleState(), gpuPod(1), nodeInfos("a"))
	if st.Code() != framework.Skip {
		t.Fatalf("want Skip, got %v", st)
	}
}

func TestPreScoreSkipsNonGPUPods(t *testing.T) {
	sim := &fakeSim{}
	st := NewWithClient(sim).PreScore(context.Background(), framework.NewCycleState(), gpuPod(0), nodeInfos("a"))
	if st.Code() != framework.Skip || sim.lastReq != nil {
		t.Fatalf("want Skip without RPC, got %v (rpc=%v)", st, sim.lastReq)
	}
}

func TestPostBindReportsToSim(t *testing.T) {
	sim := &fakeSim{}
	NewWithClient(sim).PostBind(context.Background(), nil, gpuPod(2), "node-a")
	if len(sim.binds) != 1 || sim.binds[0].Node != "node-a" || sim.binds[0].Kind != pb.PodEvent_KIND_BIND {
		t.Fatalf("unexpected binds: %v", sim.binds)
	}
}
