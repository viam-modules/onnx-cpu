// package onnx_cpu is mlmodel service module that can run ONNX files on the CPU using an external runtime environment.
package onnx_cpu

import (
	"context"
	"path"
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

// Model is the name of the module
var Model = resource.ModelNamespace("viam").WithFamily("mlmodel").WithModel("onnx-cpu")

// DataTypeMap maps the long ONNX data type labels to the data type as written in Go.
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

// Config only needs the path to the .onnx file as well as an optional path to the file of labels.
type Config struct {
	ModelPath string `json:"model_path"`
	LabelPath string `json:"label_path"`
}

// Validate makes sure that the required model path is not empty
func (cfg *Config) Validate(validatePath string) ([]string, []string, error) {
	if cfg.ModelPath == "" {
		return nil, nil, utils.NewConfigValidationFieldRequiredError(validatePath, "model_path")
	}
	ext := path.Ext(cfg.ModelPath)
	if ext != ".onnx" {
		base := path.Base(cfg.ModelPath)
		return nil, nil, errors.Errorf("model_path filename must end in .onnx. The filename is %s", base)
	}
	return nil, nil, nil
}

// modelSession stores all the relevant info needed for running the ONNX model.
// Currently this module limits the input and output tensors to be of type uint8 and/or float32.
type modelSession struct {
	Session    *ort.DynamicAdvancedSession
	InputInfo  []ort.InputOutputInfo
	OutputInfo []ort.InputOutputInfo
	InputType  ort.TensorElementDataType
	OutputType ort.TensorElementDataType
}

// onnxCPU is the struct that fulfills the interface for an ML Model Service.
type onnxCPU struct {
	resource.AlwaysRebuild
	name     resource.Name
	logger   logging.Logger
	session  modelSession
	metadata mlmodel.MLMetadata
}

// initModel will check to see if the inputs and outputs of the onnx model are valid, creates
// the metadata structure, and also starts the model session.
func initModel(name resource.Name, cfg *Config, logger logging.Logger) (*onnxCPU, error) {
	ocpu := &onnxCPU{name: name, logger: logger}
	libPath, err := getSharedLibPath()
	if err != nil {
		return nil, err
	}
	ort.SetSharedLibraryPath(libPath)
	if !ort.IsInitialized() {
		err = ort.InitializeEnvironment()
		if err != nil {
			return nil, err
		}
	}
	// get the input and output tensor info from the ONNX model
	inputInfo, outputInfo, err := ort.GetInputOutputInfo(cfg.ModelPath)
	if err != nil {
		return nil, err
	}

	// create the metadata and store the label path in the metadata if it was provided
	ocpu.metadata = createMetadata(inputInfo, outputInfo, cfg.LabelPath)

	// create the session object
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

// Name returns the name of the service.
func (ocpu *onnxCPU) Name() resource.Name {
	return ocpu.name
}

// DoCommand is unimplemented for this service.
func (ocpu *onnxCPU) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return nil, resource.ErrDoUnimplemented
}

// Metadata holds important info about the model, such as the name, shape, and data type of
// its input and output tensors.
func (ocpu *onnxCPU) Metadata(ctx context.Context) (mlmodel.MLMetadata, error) {
	return ocpu.metadata, nil
}

// Infer is where everything happens. The input data is loaded from grpc into the tensor objects,
// The model is run on the inputs to produce the output tensors,
// And the output tensors from the model are converted into protobuf compatible structs.
func (ocpu *onnxCPU) Infer(ctx context.Context, tensors ml.Tensors) (ml.Tensors, error) {
	outTensors := ml.Tensors{}
	lenInputs := len(ocpu.session.InputInfo)
	lenOutputs := len(ocpu.session.OutputInfo)
	// TODO: make this less bad, essentially have to create a case for every data type combo
	// Does the static typing really make this the only possible solution?
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

// copy the data into the onnx runtime tensors
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

// copy the data into the protobuf compatible tensors
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

func (ocpu *onnxCPU) Close(ctx context.Context) error {
	// destroy session
	err := ocpu.session.Session.Destroy()
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
