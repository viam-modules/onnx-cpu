package onnx_cpu

import (
	"context"
	"runtime"

	"github.com/pkg/errors"
	ort "github.com/yalue/onnxruntime_go"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/ml"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/mlmodel"
	"gorgonia.org/tensor"
)

var Model = resource.ModelNamespace("viam-labs").WithFamily("mlmodel").WithModel("onnx-cpu")

func init() {
	resource.RegisterService(mlmodel.API, Model, resource.Registration[mlmodel.Service, *Config]{
		Constructor: func(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (mlmodel.Service, error) {
			newConf, err := resource.NativeConfig[*Config](conf)
			if err != nil {
				return nil, err
			}
			ocpu, err := initModel(conf.ResourceName(), newConf, logger)
			if err != nil {
				return nil, err
			}
			return ocpu, nil
		},
	})
}

type Config struct {
	modelPath string `json:"model_path"`
}

func (cfg *Config) Validate(path string) ([]string, error) {
	if cfg.modelPath == "" {
		return nil, errors.New("config attribute 'model_path' must point to a onnx model file")
	}
	return nil, nil
}

type modelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[uint8]
	Output  []*ort.Tensor[float32]
}

type onnxCPU struct {
	resource.AlwaysRebuild
	name     resource.Name
	logger   logging.Logger
	session  modelSession
	metadata mlmodel.MLMetadata
}

func initModel(name resource.Name, cfg *Config, logger logging.Logger) (*onnxCPU, error) {
	ocpu := &onnxCPU{name: name, logger: logger}
	libPath, err := getSharedLibPath()
	if err != nil {
		return nil, err
	}
	ort.SetSharedLibraryPath(libPath)
	err = ort.InitializeEnvironment()
	if err != nil {
		return nil, err
	}
	// get the input and output tensor info
	inputInfo, outputInfo, err := ort.GetInputOutputInfo(cfg.modelPath)
	if err != nil {
		return nil, err
	}
	// create the metadata
	ocpu.metadata = createMetadata(inputInfo, outputInfo)
	inputNames := make([]string, 0, len(inputInfo))
	for _, in := range inputInfo {
		inputNames = append(inputNames, in.Name)
	}
	outputNames := make([]string, 0, len(outputInfo))
	for _, out := range outputInfo {
		ouputNames = append(outputNames, out.Name)
	}
	// create the session
	session, err := ort.NewDynamicAdvancedSession(cfg.modelPath,
		inputNames, outputNames, options,
	)

	modelSes := modelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensors,
	}
	ocpu.session = modelSes

	return ocpu, nil
}

func (ocpu *onnxCPU) Name() resource.Name {
	return ocpu.name
}

func (ocpu *onnxCPU) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return nil, resource.ErrDoUnimplemented
}

func (ocpu *onnxCPU) Infer(ctx context.Context, tensors ml.Tensors) (ml.Tensors, error) {
	input, err := processInput(tensors)
	if err != nil {
		return nil, err
	}
	inTensor := ocpu.session.Input.GetData()
	copy(inTensor, input)
	err = ocpu.session.Session.Run()
	if err != nil {
		return nil, err
	}
	outputData := make([][]float32, 0, 8)
	for _, out := range ocpu.session.Output {
		outputData = append(outputData, out.GetData())
	}
	return processOutput(outputData)
}

func processInput(tensors ml.Tensors) ([]uint8, error) {
	var imageTensor *tensor.Dense
	// if length of tensors is 1, just grab the first tensor
	// if more than 1 grab the one called input tensor, or image
	if len(tensors) == 1 {
		for _, t := range tensors {
			imageTensor = t
			break
		}
	} else {
		for name, t := range tensors {
			if name == "image" || name == "input_tensor" {
				imageTensor = t
				break
			}
		}
	}
	if imageTensor == nil {
		return nil, errors.New("no valid input tensor called 'image' or 'input_tensor' found")
	}
	if uint8Data, ok := imageTensor.Data().([]uint8); ok {
		return uint8Data, nil
	}
	return nil, errors.Errorf("input tensor must be of tensor type UIn8, got %v", imageTensor.Dtype())
}

func processOutput(outputs [][]float32) (ml.Tensors, error) {
	// there are 8 output tensors. Turn them into tensors with the right backing
	outMap := ml.Tensors{}
	outMap["detection_anchor_indices"] = tensor.New(
		tensor.WithShape(1, 100),
		tensor.WithBacking(outputs[0]),
	)
	outMap["location"] = tensor.New(
		tensor.WithShape(1, 100, 4),
		tensor.WithBacking(outputs[1]),
	)
	outMap["category"] = tensor.New(
		tensor.WithShape(1, 100),
		tensor.WithBacking(outputs[2]),
	)
	outMap["detection_multiclass_scores"] = tensor.New(
		tensor.WithShape(1, 100, 2),
		tensor.WithBacking(outputs[3]),
	)
	outMap["score"] = tensor.New(
		tensor.WithShape(1, 100),
		tensor.WithBacking(outputs[4]),
	)
	outMap["num_detections"] = tensor.New(
		tensor.WithShape(1),
		tensor.WithBacking(outputs[5]),
	)
	outMap["raw_detection_boxes"] = tensor.New(
		tensor.WithShape(1, 1917, 4),
		tensor.WithBacking(outputs[6]),
	)
	outMap["raw_detection_scores"] = tensor.New(
		tensor.WithShape(1, 1917, 2),
		tensor.WithBacking(outputs[7]),
	)
	return outMap, nil
}

func (ocpu *onnxCPU) Metadata(ctx context.Context) (mlmodel.MLMetadata, error) {
	return ocpu.metadata, nil
}

func (ocpu *onnxCPU) Close(ctx context.Context) error {
	// destroy session
	err := ocpu.session.Session.Destroy()
	if err != nil {
		return err
	}
	// destroy tensors
	for _, outTensor := range ocpu.session.Output {
		err = outTensor.Destroy()
		if err != nil {
			return err
		}
	}
	err = ocpu.session.Input.Destroy()
	if err != nil {
		return err
	}
	// destroy environment
	err = ort.DestroyEnvironment()
	if err != nil {
		return err
	}
	return nil
}

func getSharedLibPath() (string, error) {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime.dll", nil
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.dylib", nil
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.so", nil
		}
		return "./third_party/onnxruntime.so", nil
	}
	return "", errors.Errorf("Unable to find a version of the onnxruntime library supporting %s %s", runtime.GOOS, runtime.GOARCH)
}

func createMetadata(inputInfo, outputInfo []ort.InputOutputInfo) mlmodel.MLMetadata {
	md := mlmodel.MLMetadata{}
	md.ModelName = "onnx_model"
	// inputs
	inputs := []mlmodel.TensorInfo{}
	for _, in := range inputInfo {
		shape := convertInt64SliceToInt(in.Dimensions)
		info := mlmodel.TensorInfo{
			Name:     in.Name,
			DataType: in.DataType.String(),
			Shape:    shape,
		}
		inputs = append(inputs, info)
	}
	md.Inputs = inputs
	// outputs
	outputs := []mlmodel.TensorInfo{}
	for _, out := range outputInfo {
		shape := convertInt64SliceToInt(out.Dimensions)
		info := mlmodel.TensorInfo{
			Name:     out.Name,
			DataType: out.DataType.String(),
			Shape:    shape,
		}
		outputs = append(outputs, info)
	}
	md.Outputs = outputs
	return md
}

func convertInt64SliceToInt(sliceInt64 []int64) []int {
	sliceInt := make([]int, 0, len(sliceInt64))
	for _, v := range sliceInt64 {
		sliceInt = append(sliceInt, int(v))
	}
	return sliceInt
}
