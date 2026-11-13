// Command isotherm-scheduler is kube-scheduler with the Isotherm plugin compiled in.
//
//	isotherm-scheduler --config deploy/k8s/scheduler-config.yaml
package main

import (
	"os"

	"k8s.io/component-base/cli"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"

	"isotherm/scheduler/pkg/plugin"
)

func main() {
	cmd := app.NewSchedulerCommand(app.WithPlugin(plugin.Name, plugin.New))
	os.Exit(cli.Run(cmd))
}
