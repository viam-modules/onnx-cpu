package onnx_cpu

import (
	"context"
	"runtime"
	"strings"

	"github.com/pkg/errors"
	ort "github.com/yalue/onnxruntime_go"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/ml"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/mlmodel"
	"go.viam.com/utils"
	"gorgonia.org/tensor"
)

var Model = resource.ModelNamespace("viam-labs").WithFamily("mlmodel").WithModel("onnx-cpu")

var DataTypeMap = map[ort.TensorElementDataType]string{
	ort.TensorElementDataTypeFloat: "float32",
	ort.TensorElementDataTypeUint8: "uint8",
}

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
	ModelPath string `json:"model_path"`
	LabelPath string `json:"label_path"`
}

func (cfg *Config) Validate(path string) ([]string, error) {
	if cfg.ModelPath == "" {
		return nil, utils.NewConfigValidationFieldRequiredError(path, "model_path")
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
	inputInfo, outputInfo, err := ort.GetInputOutputInfo(cfg.ModelPath)
	if err != nil {
		return nil, err
	}
	// create the metadata
	ocpu.metadata = createMetadata(inputInfo, outputInfo, cfg.LabelPath)
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
	session, err := ort.NewDynamicAdvancedSession(cfg.ModelPath,
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
	lenInputs := len(ocpu.session.InputInfo)
	lenOutputs := len(ocpu.session.OutputInfo)
	// TODO: make this less bad, is it really only possible by doing a type switch?
	switch ocpu.session.InputType {
	case ort.TensorElementDataTypeFloat:
		inputs := make([]*ort.Tensor[float32], 0, lenInputs)
		inputs, err := mlTensorsToOnnxTensors(tensors, inputs, ocpu.session.InputInfo)
		if err != nil {
			return nil, err
		}
		defer func() {
			utils.UncheckedError(destroyTensors(inputs))
		}()
		switch ocpu.session.OutputType {
		case ort.TensorElementDataTypeFloat:
			outputs := make([]*ort.Tensor[float32], 0, lenOutputs)
			outputs, err := runModel(ocpu.session.Session, lenOutputs, inputs, outputs)
			if err != nil {
				return nil, err
			}
			defer func() {
				utils.UncheckedError(destroyTensors(outputs))
			}()
			err = onnxTensorsToMlTensors(outputs, outTensors, ocpu.session.OutputInfo)
			if err != nil {
				return nil, err
			}
		case ort.TensorElementDataTypeUint8:
			outputs := make([]*ort.Tensor[uint8], 0, lenOutputs)
			outputs, err := runModel(ocpu.session.Session, lenOutputs, inputs, outputs)
			if err != nil {
				return nil, err
			}
			defer func() {
				utils.UncheckedError(destroyTensors(outputs))
			}()
			err = onnxTensorsToMlTensors(outputs, outTensors, ocpu.session.OutputInfo)
			if err != nil {
				return nil, err
			}
		default:
			return nil, errors.Errorf("output tensor type %s not implemented", ocpu.session.OutputType.String())
		}
	case ort.TensorElementDataTypeUint8:
		inputs := make([]*ort.Tensor[uint8], 0, lenOutputs)
		inputs, err := mlTensorsToOnnxTensors(tensors, inputs, ocpu.session.InputInfo)
		if err != nil {
			return nil, err
		}
		defer func() {
			utils.UncheckedError(destroyTensors(inputs))
		}()
		switch ocpu.session.OutputType {
		case ort.TensorElementDataTypeFloat:
			outputs := make([]*ort.Tensor[float32], 0, lenOutputs)
			outputs, err := runModel(ocpu.session.Session, lenOutputs, inputs, outputs)
			if err != nil {
				return nil, err
			}
			defer func() {
				utils.UncheckedError(destroyTensors(outputs))
			}()
			err = onnxTensorsToMlTensors[float32](outputs, outTensors, ocpu.session.OutputInfo)
			if err != nil {
				return nil, err
			}
		case ort.TensorElementDataTypeUint8:
			outputs := make([]*ort.Tensor[uint8], 0, lenOutputs)
			outputs, err := runModel(ocpu.session.Session, lenOutputs, inputs, outputs)
			if err != nil {
				return nil, err
			}
			defer func() {
				utils.UncheckedError(destroyTensors(outputs))
			}()
			err = onnxTensorsToMlTensors[uint8](outputs, outTensors, ocpu.session.OutputInfo)
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

func runModel[M, N ort.TensorData](session *ort.DynamicAdvancedSession, outputLen int, inputs []*ort.Tensor[M], outputs []*ort.Tensor[N]) ([]*ort.Tensor[N], error) {
	arbIn := toArbitraryTensor(inputs)
	arbOut := make([]ort.ArbitraryTensor, outputLen)
	err := session.Run(arbIn, arbOut)
	if err != nil {
		return nil, errors.Wrap(err, "failed to Run on Infer command")
	}
	for _, out := range arbOut {
		o, ok := out.(*ort.Tensor[N])
		if !ok {
			return nil, errors.Errorf("could not convert output tensor from Run to type %T", o)
		}
		outputs = append(outputs, o)
	}
	return outputs, nil
}

func toArbitraryTensor[T ort.TensorData](in []*ort.Tensor[T]) []ort.ArbitraryTensor {
	out := make([]ort.ArbitraryTensor, 0, len(in))
	for _, t := range in {
		out = append(out, t)
	}
	return out
}

func destroyTensors[T ort.TensorData](toDestroy []*ort.Tensor[T]) error {
	for _, t := range toDestroy {
		err := t.Destroy()
		if err != nil {
			return err
		}
	}
	return nil
}

// copy the data into the input tensors
func mlTensorsToOnnxTensors[T ort.TensorData](tensors ml.Tensors, inputs []*ort.Tensor[T], info []ort.InputOutputInfo) ([]*ort.Tensor[T], error) {
	// order is given by InputInfo array. The names must match
	for _, inf := range info {
		denseTensor, found := tensors[inf.Name]
		if !found {
			return nil, errors.Errorf("input tensor with name %q is required", inf.Name)
		}
		typedDenseData, ok := denseTensor.Data().([]T)
		if !ok {
			return nil, errors.Errorf("input tensor %s is of type %v, not %s", inf.Name, denseTensor.Dtype(), inf.DataType.String())
		}
		shape := ort.Shape{}
		for _, s := range denseTensor.Shape() {
			shape = append(shape, int64(s))
		}
		input, err := ort.NewTensor(shape, typedDenseData)
		if err != nil {
			return nil, errors.Wrapf(err, "input tensor %s encountered an error", inf.Name)
		}
		inputs = append(inputs, input)
	}
	return inputs, nil
}

func onnxTensorsToMlTensors[T ort.TensorData](outputs []*ort.Tensor[T], tensors ml.Tensors, info []ort.InputOutputInfo) error {
	for i, inf := range info {
		t := outputs[i]
		shape := make([]int, 0, len(t.GetShape()))
		for _, d := range t.GetShape() {
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
	switch arch := strings.Join([]string{runtime.GOOS, runtime.GOARCH}, "-"); arch {
	case "windows-amd64":
		return "./third_party/onnxruntime.dll", nil
	case "darwin-arm64":
		return "./third_party/onnxruntime_arm64.dylib", nil
	case "linux-arm64":
		return "./third_party/onnxruntime_arm64.so", nil
	case "linux-amd64":
		return "./third_party/onnxruntime.so", nil
	case "android-386":
		return "./third_party/onnx-android-x86.so", nil
	case "android-arm64":
		return "./third_party/onnx-android-arm64-v8a.so", nil
	}
	return "", errors.Errorf("Unable to find a version of the onnxruntime library supporting %s %s", runtime.GOOS, runtime.GOARCH)
}

func createMetadata(inputInfo, outputInfo []ort.InputOutputInfo, labelPath string) mlmodel.MLMetadata {
	md := mlmodel.MLMetadata{}
	md.ModelName = "onnx_model"
	// inputs
	inputs := []mlmodel.TensorInfo{}
	for _, in := range inputInfo {
		shape := convertInt64SliceToInt(in.Dimensions)
		dataType := in.DataType.String()
		if dataTypeString, ok := DataTypeMap[in.DataType]; ok {
			dataType = dataTypeString
		}
		info := mlmodel.TensorInfo{
			Name:     in.Name,
			DataType: dataType,
			Shape:    shape,
		}
		inputs = append(inputs, info)
	}
	md.Inputs = inputs
	// outputs
	outputs := []mlmodel.TensorInfo{}
	for _, out := range outputInfo {
		shape := convertInt64SliceToInt(out.Dimensions)
		dataType := out.DataType.String()
		if dataTypeString, ok := DataTypeMap[out.DataType]; ok {
			dataType = dataTypeString
		}
		extra := map[string]interface{}{}
		extra["labels"] = labelPath // put label path info in the Extra field
		info := mlmodel.TensorInfo{
			Name:     out.Name,
			DataType: dataType,
			Shape:    shape,
			Extra:    extra,
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
