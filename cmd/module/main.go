package main

import (
	"context"

	onnx "github.com/viam-labs/onnx-cpu"
	ort "github.com/yalue/onnxruntime_go"
	"go.viam.com/rdk/module"
	"go.viam.com/rdk/services/mlmodel"
)

func main() {
	err := realMain()
	if err != nil {
		panic(err)
	}
}
func realMain() error {

	ctx := context.Background()
	logger := logging.NewDebugLogger("client")

	myMod, err := module.NewModuleFromArgs(ctx)
	if err != nil {
		return err
	}

	err = myMod.AddModelFromRegistry(ctx, mlmodel.API, onnx.Model)
	if err != nil {
		return err
	}

	err = myMod.Start(ctx)
	defer myMod.Close(ctx)
	if err != nil {
		return err
	}
	defer onnxClose(ctx) //nolint:errcheck
	<-ctx.Done()
	return nil
}

func onnxClose(ctc context.Context) error {
	// destroy environment
	if ort.IsInitialized() {
		err := ort.DestroyEnvironment()
		if err != nil {
			return err
		}
	}
	return nil
}
