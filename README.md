# onnx-cpu

## Build it

if you need to package it up in a module, do the following

```
go build -o module cmd/module/main.go
chmod a+x module
tar -czf module.tar.gz module third_party/ ir_mobilenet.onnx
```

`third_party` contains all of the `.so` files for the ONNX runntime. You can package only the one you need.

For Android, the Makefile will create the tar file.

You can then locally run the module. the triplet is `viam-labs:mlmodel:onnx-cpu`

## Config

There is only one attribute you need for onnx-cpu, `model_path`, the full path to the ONNX file.

```
    {
      "name": "onnx",
      "type": "mlmodel",
      "model": "viam-labs:mlmodel:onnx-cpu",
      "attributes": {
        "model_path": "/path/to/onnx_file/detector_googlenet.onnx"
      }
    }
```

## Use it with a vision service

You will most likely need to rename the input and outputs tensors coming from the ONNX file. There is a new attribute on the mlmodel vision service that allows you do that. You have to go to `RAW JSON` to actually be able to input the remapping, though

```
   {
      "type": "vision",
      "model": "mlmodel",
      "attributes": {
        "mlmodel_name": "onnx",
        "remap_output_names": {
          "detection_classes": "category",
          "detection_boxes": "location",
          "detection_scores": "score"
        },
        "remap_input_names": {
          "input_tensor": "image"
        }
      },
      "name": "onnx-vision"
    }
```
