// Package plugin implements Isotherm, a kube-scheduler plugin that ranks nodes by the
// thermal headroom the simulator predicts for a pod.
//
// The plugin never computes temperatures itself: PreScore sends one ScoreCandidates RPC
// per pod (all candidate nodes at once) and Score only reads the answer back. If the
// simulator is unreachable the plugin skips itself and other score plugins decide.
package plugin

import (
	"context"
	"fmt"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"

	pb "isotherm/scheduler/gen/isothermpb"
)

const (
	// Name is the plugin name used in KubeSchedulerConfiguration.
	Name = "Isotherm"

	// SimAddrEnv overrides the simulator address.
	SimAddrEnv     = "ISOTHERM_SIM_ADDR"
	defaultSimAddr = "localhost:50051"
	rpcTimeout     = 2 * time.Second

	stateKey framework.StateKey = Name + "/scores"
)

// Isotherm is the scheduler plugin.
type Isotherm struct {
	sim pb.ThermalSimClient
}

var (
	_ framework.PreScorePlugin  = &Isotherm{}
	_ framework.ScorePlugin     = &Isotherm{}
	_ framework.ScoreExtensions = &Isotherm{}
	_ framework.PostBindPlugin  = &Isotherm{}
)

// New is the plugin factory registered with the scheduler.
func New(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
	addr := os.Getenv(SimAddrEnv)
	if addr == "" {
		addr = defaultSimAddr
	}
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("connect to thermal sim at %s: %w", addr, err)
	}
	klog.InfoS("Isotherm plugin connected", "sim", addr)
	return NewWithClient(pb.NewThermalSimClient(conn)), nil
}

// NewWithClient builds the plugin around an existing client (used by tests).
func NewWithClient(c pb.ThermalSimClient) *Isotherm { return &Isotherm{sim: c} }

// Name implements framework.Plugin.
func (p *Isotherm) Name() string { return Name }

type cycleScores struct {
	byNode map[string]*pb.CandidateScore
}

// Clone implements framework.StateData. Scores are read-only after PreScore.
func (s *cycleScores) Clone() framework.StateData { return s }

// PreScore asks the simulator to score every candidate node in a single RPC.
func (p *Isotherm) PreScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodes []*framework.NodeInfo) *framework.Status {
	w := WorkloadFromPod(pod)
	if w.GpuCount == 0 {
		return framework.NewStatus(framework.Skip) // not a GPU pod; nothing thermal to decide
	}
	names := make([]string, 0, len(nodes))
	for _, n := range nodes {
		names = append(names, n.Node().Name)
	}

	ctx, cancel := context.WithTimeout(ctx, rpcTimeout)
	defer cancel()
	resp, err := p.sim.ScoreCandidates(ctx, &pb.ScoreRequest{Workload: w, CandidateNodes: names})
	if err != nil {
		klog.ErrorS(err, "Thermal sim unavailable, skipping thermal scoring", "pod", klog.KObj(pod))
		return framework.NewStatus(framework.Skip)
	}

	byNode := make(map[string]*pb.CandidateScore, len(resp.Scores))
	for _, s := range resp.Scores {
		byNode[s.Node] = s
	}
	state.Write(stateKey, &cycleScores{byNode: byNode})
	return nil
}

// Score returns the simulator's answer for one node as a raw score; see RawScore.
func (p *Isotherm) Score(_ context.Context, state *framework.CycleState, _ *v1.Pod, nodeName string) (int64, *framework.Status) {
	data, err := state.Read(stateKey)
	if err != nil {
		return 0, framework.AsStatus(fmt.Errorf("reading %s: %w", stateKey, err))
	}
	return RawScore(data.(*cycleScores).byNode[nodeName]), nil
}

// ScoreExtensions implements framework.ScorePlugin.
func (p *Isotherm) ScoreExtensions() framework.ScoreExtensions { return p }

// NormalizeScore maps raw scores onto [0, framework.MaxNodeScore].
func (p *Isotherm) NormalizeScore(_ context.Context, _ *framework.CycleState, _ *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	raw := make([]int64, len(scores))
	for i := range scores {
		raw[i] = scores[i].Score
	}
	for i, s := range Normalize(raw, framework.MaxNodeScore) {
		scores[i].Score = s
	}
	return nil
}

// PostBind tells the simulator where the pod landed so its load matches the cluster.
func (p *Isotherm) PostBind(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeName string) {
	w := WorkloadFromPod(pod)
	if w.GpuCount == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, rpcTimeout)
	defer cancel()
	resp, err := p.sim.ApplyEvent(ctx, &pb.PodEvent{Kind: pb.PodEvent_KIND_BIND, Workload: w, Node: nodeName})
	switch {
	case err != nil:
		klog.ErrorS(err, "Failed to report bind to thermal sim", "pod", klog.KObj(pod), "node", nodeName)
	case !resp.Ok:
		klog.ErrorS(nil, "Thermal sim rejected bind", "pod", klog.KObj(pod), "node", nodeName, "reason", resp.Error)
	}
}
