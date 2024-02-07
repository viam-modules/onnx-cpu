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
	Session    *ort.DynamicAdvancedSession
	InputInfo  []ort.InputOutputInfo
	OutputInfo []ort.InputOutputInfo
	InputType  ort.TensorElementDataType
	OutputType ort.TensorElementDataType
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
	// create the inputs and outputs
	// input
	inputNames := make([]string, 0, len(inputInfo))
	var inputType ort.TensorElementDataType
	if len(inputInfo) != 0 {
		inputType = inputInfo[0].DataType
		if inputType != ort.TensorElementDataTypeFloat && inputType != ort.TensorElementDataTypeUint8 {
			return nil, errors.Errorf("currently only supporting input tensors of type uint8 or float32, got %s", inputType)
		}
	}
	for _, in := range inputInfo {
		if in.DataType != inputType {
			return nil, errors.New("all input tensors must be of the same data type, mixing data types not currently supported.")
		}
		inputNames = append(inputNames, in.Name)
	}
	// output
	outputNames := make([]string, 0, len(outputInfo))
	var outputType ort.TensorElementDataType
	outputType = ort.TensorElementDataTypeUndefined
	if len(outputInfo) != 0 {
		outputType = outputInfo[0].DataType
		if outputType != ort.TensorElementDataTypeFloat && inputType != ort.TensorElementDataTypeUint8 {
			return nil, errors.Errorf("currently only supporting output tensors of type uint8 or float32, got %s", inputType)
		}
	}
	for _, out := range outputInfo {
		if out.DataType != outputType {
			return nil, errors.New("all output tensors must be of the same data type, mixing data types not currently supported.")
		}
		outputNames = append(outputNames, out.Name)
	}
	// create the session
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	session, err := ort.NewDynamicAdvancedSession(cfg.modelPath,
		inputNames, outputNames, options,
	)
	if err != nil {
		return nil, err
	}

	modelSes := modelSession{
		Session:    session,
		InputInfo:  inputInfo,
		OutputInfo: outputInfo,
		InputType:  inputType,
		OutputType: outputType,
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
	outTensors := ml.Tensors{}
	// TODO: make this less bad, is it really only possible by doing a type switch?
	switch ocpu.session.InputType {
	case ort.TensorElementDataTypeFloat:
		inputs := make([]*ort.Tensor[float32], 0, len(ocpu.session.InputInfo))
		err := mlTensorsToOnnxTensors(tensors, inputs, ocpu.session.InputInfo)
		if err != nil {
			return nil, err
		}
		switch ocpu.session.OutputType {
		case ort.TensorElementDataTypeFloat:
			outputs := make([]*ort.Tensor[float32], 0, len(ocpu.session.OutputInfo))
			arbIn := toArbitraryTensor(inputs)
			arbOut := toArbitraryTensor(outputs)
			err = ocpu.session.Session.Run(arbIn, arbOut)
			if err != nil {
				return nil, err
			}
			err := onnxTensorsToMlTensors(outputs, outTensors, ocpu.session.OutputInfo)
			if err != nil {
				return nil, err
			}
		case ort.TensorElementDataTypeUint8:
			outputs := make([]*ort.Tensor[uint8], 0, len(ocpu.session.OutputInfo))
			arbIn := toArbitraryTensor(inputs)
			arbOut := toArbitraryTensor(outputs)
			err = ocpu.session.Session.Run(arbIn, arbOut)
			if err != nil {
				return nil, err
			}
			err := onnxTensorsToMlTensors(outputs, outTensors, ocpu.session.OutputInfo)
			if err != nil {
				return nil, err
			}
		default:
			return nil, errors.Errorf("output tensor type %s not implemented", ocpu.session.OutputType.String())
		}
	case ort.TensorElementDataTypeUint8:
		inputs := make([]*ort.Tensor[uint8], 0, len(ocpu.session.InputInfo))
		err := mlTensorsToOnnxTensors(tensors, inputs, ocpu.session.InputInfo)
		if err != nil {
			return nil, err
		}
		switch ocpu.session.OutputType {
		case ort.TensorElementDataTypeFloat:
			outputs := make([]*ort.Tensor[float32], 0, len(ocpu.session.OutputInfo))
			arbIn := toArbitraryTensor(inputs)
			arbOut := toArbitraryTensor(outputs)
			err = ocpu.session.Session.Run(arbIn, arbOut)
			if err != nil {
				return nil, err
			}
			err := onnxTensorsToMlTensors[float32](outputs, outTensors, ocpu.session.OutputInfo)
			if err != nil {
				return nil, err
			}
		case ort.TensorElementDataTypeUint8:
			outputs := make([]*ort.Tensor[uint8], 0, len(ocpu.session.OutputInfo))
			arbIn := toArbitraryTensor(inputs)
			arbOut := toArbitraryTensor(outputs)
			err = ocpu.session.Session.Run(arbIn, arbOut)
			if err != nil {
				return nil, err
			}
			err := onnxTensorsToMlTensors[uint8](outputs, outTensors, ocpu.session.OutputInfo)
			if err != nil {
				return nil, err
			}
		default:
			return nil, errors.Errorf("output tensor type %s not implemented", ocpu.session.OutputType.String())
		}
	default:
		return nil, errors.Errorf("input tensor type %s not implemented", ocpu.session.InputType.String())
	}
	return outTensors, nil
}

func toArbitraryTensor[T ort.TensorData](in []*ort.Tensor[T]) []ort.ArbitraryTensor {
	out := make([]ort.ArbitraryTensor, 0, len(in))
	for _, t := range in {
		out = append(out, t)
	}
	return out
}

// copy the data into the input tensors
func mlTensorsToOnnxTensors[T ort.TensorData](tensors ml.Tensors, inputs []*ort.Tensor[T], info []ort.InputOutputInfo) error {
	// order is given by InputInfo array. The names must match
	for _, inf := range info {
		denseTensor, found := tensors[inf.Name]
		if !found {
			return errors.Errorf("input tensor with name %q is required", inf.Name)
		}
		typedDenseData, ok := denseTensor.Data().([]T)
		if !ok {
			return errors.Errorf("input tensor %s is of type %v, not %s", inf.Name, denseTensor.Dtype(), inf.DataType.String())
		}
		input, err := ort.NewTensor(inf.Dimensions, typedDenseData)
		if err != nil {
			return errors.Wrapf(err, "input tensor %s encountered an error", inf.Name)
		}
		inputs = append(inputs, input)
	}
	return nil
}

func onnxTensorsToMlTensors[T ort.TensorData](outputs []*ort.Tensor[T], tensors ml.Tensors, info []ort.InputOutputInfo) error {
	for i, inf := range info {
		t := outputs[i]
		shape := make([]int, 0, len(inf.Dimensions))
		for _, d := range inf.Dimensions {
			shape = append(shape, int(d))
		}
		tensors[inf.Name] = tensor.New(
			tensor.WithShape(shape...),
			tensor.WithBacking(t.GetData()),
		)
	}
	return nil
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
