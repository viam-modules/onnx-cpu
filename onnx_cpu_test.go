package onnx_cpu

import (
	"context"
	"testing"

	"github.com/nfnt/resize"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/ml"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/rimage"
	"go.viam.com/rdk/services/mlmodel"
	"go.viam.com/test"
	"gorgonia.org/tensor"
)

func TestImageClassification(t *testing.T) {
	logger := logging.NewTestLogger(t)
	name := resource.NewName(mlmodel.API, "test_model")
	cfg := &Config{"./test_files/age_googlenet.onnx", ""}
	theModel, err := initModel(name, cfg, logger)
	test.That(t, err, test.ShouldBeNil)

	// check the metaata
	md, err := theModel.Metadata(context.Background())
	test.That(t, err, test.ShouldBeNil)
	test.That(t, md.ModelName, test.ShouldEqual, "onnx_model")
	test.That(t, len(md.Inputs), test.ShouldEqual, 1)
	test.That(t, len(md.Outputs), test.ShouldEqual, 1)
	test.That(t, md.Inputs[0].Shape, test.ShouldResemble, []int{1, 3, 224, 224})
	test.That(t, md.Outputs[0].Shape, test.ShouldResemble, []int{1, 8})
	test.That(t, md.Outputs[0].Extra, test.ShouldContainKey, "labels")

	// check infer with a test image
	img, err := rimage.NewImageFromFile("./test_files/person2.jpeg")
	test.That(t, err, test.ShouldBeNil)
	resized := resize.Resize(224, 224, img, resize.Bilinear)
	// create input tensor
	inMap := ml.Tensors{}
	inMap["input"] = tensor.New(
		tensor.WithShape(1, resized.Bounds().Dy(), resized.Bounds().Dx(), 3),
		tensor.WithBacking(rimage.ImageToFloatBuffer(resized)),
	)
	err = inMap["input"].T(0, 3, 1, 2)
	test.That(t, err, test.ShouldBeNil)
	err = inMap["input"].Transpose()
	test.That(t, err, test.ShouldBeNil)
	// infer
	outMap, err := theModel.Infer(context.Background(), inMap)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(outMap), test.ShouldEqual, 1)
	test.That(t, outMap, test.ShouldContainKey, "loss3/loss3_Y")
	expectedShape := tensor.Shape{1, 8}
	test.That(t, outMap["loss3/loss3_Y"].Shape(), test.ShouldResemble, expectedShape)
	// does the first detection have a confidence more than 90%
	detScore, err := outMap["loss3/loss3_Y"].At(0, 4)
	test.That(t, err, test.ShouldBeNil)
	score, ok := detScore.(float32)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, score, test.ShouldBeGreaterThan, 0.8)
	err = theModel.Close(context.Background())
	test.That(t, err, test.ShouldBeNil)
}

func TestImageDetection(t *testing.T) {
	logger := logging.NewTestLogger(t)
	name := resource.NewName(mlmodel.API, "test_model")
	cfg := &Config{"./test_files/ir_mobilenet.onnx", "/path/to/labels.txt"}
	theModel, err := initModel(name, cfg, logger)
	test.That(t, err, test.ShouldBeNil)

	// check the metaata
	md, err := theModel.Metadata(context.Background())
	test.That(t, err, test.ShouldBeNil)
	test.That(t, md.ModelName, test.ShouldEqual, "onnx_model")
	test.That(t, len(md.Inputs), test.ShouldEqual, 1)
	test.That(t, len(md.Outputs), test.ShouldEqual, 8)
	test.That(t, md.Outputs[0].Extra, test.ShouldContainKey, "labels")

	// check infer with a test image
	img, err := rimage.NewImageFromFile("./test_files/person.jpeg")
	test.That(t, err, test.ShouldBeNil)
	// reshape to 300 x 300
	resized := resize.Resize(300, 300, img, resize.Bilinear)
	// create input tensor
	inMap := ml.Tensors{}
	inMap["input_tensor"] = tensor.New(
		tensor.WithShape(1, resized.Bounds().Dy(), resized.Bounds().Dx(), 3),
		tensor.WithBacking(rimage.ImageToUInt8Buffer(resized)),
	)
	// infer
	outMap, err := theModel.Infer(context.Background(), inMap)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(outMap), test.ShouldEqual, 8)
	test.That(t, outMap, test.ShouldContainKey, "detection_scores")
	expectedShape := tensor.Shape{1, 100}
	test.That(t, outMap["detection_scores"].Shape(), test.ShouldResemble, expectedShape)
	// does the first detection have a confidence more than 90%
	detScore, err := outMap["detection_scores"].At(0, 0)
	test.That(t, err, test.ShouldBeNil)
	score, ok := detScore.(float32)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, score, test.ShouldBeGreaterThan, 0.9)
	// does the second detection has a confidence of less than .5
	detScore, err = outMap["detection_scores"].At(0, 1)
	test.That(t, err, test.ShouldBeNil)
	score, ok = detScore.(float32)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, score, test.ShouldBeLessThan, 0.5)
	err = theModel.Close(context.Background())
	test.That(t, err, test.ShouldBeNil)
}
