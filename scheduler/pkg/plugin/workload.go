package plugin

import (
	"strconv"

	v1 "k8s.io/api/core/v1"

	pb "isotherm/scheduler/gen/isothermpb"
)

const (
	// GPUResource is the extended resource that counts GPUs.
	GPUResource v1.ResourceName = "nvidia.com/gpu"
	// AnnotationGPUUtil is the expected per-GPU utilization in [0, 1].
	AnnotationGPUUtil = "isotherm.io/gpu-util"
	// AnnotationDuration is the expected runtime in seconds.
	AnnotationDuration = "isotherm.io/duration-s"

	defaultGPUUtil = 0.8
)

// WorkloadID is the id shared by every component: "<namespace>/<name>".
func WorkloadID(pod *v1.Pod) string { return pod.Namespace + "/" + pod.Name }

// WorkloadFromPod translates a pod into the contract's Workload message.
func WorkloadFromPod(pod *v1.Pod) *pb.Workload {
	var gpus int64
	for _, c := range pod.Spec.Containers {
		if q, ok := c.Resources.Limits[GPUResource]; ok {
			gpus += q.Value()
		}
	}
	return &pb.Workload{
		WorkloadId: WorkloadID(pod),
		GpuCount:   int32(gpus),
		GpuUtil:    annotationFloat(pod, AnnotationGPUUtil, defaultGPUUtil),
		DurationS:  annotationFloat(pod, AnnotationDuration, 0),
	}
}

func annotationFloat(pod *v1.Pod, key string, def float64) float64 {
	if v, ok := pod.Annotations[key]; ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}
