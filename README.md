# `onnx-cpu` modular resource

This module allows you to deploy ONNX models which you can then use with the Viam `mlmodel` vision service on Android.

latest version uses onnxruntime 1.23.2

## Requirements

Before configuring your ML model, you must [create a machine](https://docs.viam.com/fleet/machines/#add-a-new-machine).

## Build and run

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/configure/#add-a-modular-resource-from-the-viam-registry) and select the `viam:mlmodel:onnx-cpu` model from the [`onnx-cpu` module](https://github.com/viam-modules/onnx-cpu).

## Configure your `onnx-cpu` ML model

Navigate to the **Config** tab of your machine's page in [the Viam app](https://app.viam.com/).
Click on the **Services** subtab and click **Create service**.
Select the `mlmodel` type, then select the `mlmodel:onnx-cpu` model.
Click **Add module**, then enter a name for your ML model and click **Create**.

On the new component panel, deploy a model on to your machine by selecting an onnx model from the **Model** dropdown or by specifying a path to an existing model on your machine.

If you prefer to configure your service using JSON, use the following attributes **attributes**:

```json
{
  "model_path": "/path/to/onnx_file/detector_googlenet.onnx",
  "label_path": "/path/to/labels.txt",
  "num_threads": 1,
  "package_reference": "viam-soleng/face-detector-onnx"
}
```

> [!NOTE]
> For more information, see [Configure a Machine](https://docs.viam.com/manage/configuration/).

### Attributes

The following attributes are available for the `viam:mlmodel:onnx-cpu` ML model service:

| Name    | Type   | Inclusion    | Description |
| ------- | ------ | ------------ | ----------- |
| `model_path` | string | **Required** | The full path to the ONNX file. |
| `label_path` | string | Optional | The full path to the names of the category labels. |
| `num_threads` | int | Optional | The number of threads. Must be 1. |
| `package_reference` | string | Optional | The package of the model to deploy if the model was deployed through the registry. For example: "viam-soleng/face-detector-onnx". |


### Example configuration

```json
    {
      "name": "onnx",
      "type": "mlmodel",
      "model": "viam:mlmodel:onnx-cpu",
      "attributes": {
        "model_path": "/path/to/onnx_file/detector_googlenet.onnx"
        "label_path": "/path/to/labels.txt"
      }
    }
```

### Configure a vision service

The module allows you to deploy an ONNX ML model with the ML model service it provides. Viam's vision service allows you to make use of the deployed detection or classification model.

Navigate to the **Config** tab of your machine's page in [the Viam app](https://app.viam.com/).
Click on the **Services** subtab and click **Create service**.
Select the `vision` type, then select the `mlmodel` model.
Enter a name for your ML model and click **Create**.

On the new service panel, select the `onnx-cpu` model you configured.

You will most likely need to [rename the input and outputs tensors](https://docs.viam.com/ml/vision/mlmodel/#tensor-names) coming from the ONNX file to use the tensor names that the vision service requires. To rename the tensors, go to your Raw JSON configuration and add the `remap_output_names` and `remap_input_names` fields to the attributes of your vision service config. For more information see [the documentation for remapping tensor names](https://docs.viam.com/ml/vision/mlmodel/#tensor-names).

Here is an example:

```json
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

Save your config, before testing your vision service.

### Next steps

You have now configured a vision service to use an ONNX ML model.
Follow these steps to [test your `mlmodel` vision service](https://docs.viam.com/ml/vision/mlmodel/#test-your-detector-or-classifier).

## Local Development

This module is written in Go. If you need to package it up into a binary to create a module, run the following commands:

```bash
go build -o module cmd/module/main.go
chmod a+x module
tar -czf module.tar.gz module third_party/
```

`third_party` contains all of the `.so` files for the ONNX runtime. You can package only the one you need.

For Android, the Makefile will create the tar file.

You can then add the module to app.viam.com locally. the model triplet is `viam:mlmodel:onnx-cpu`.

## License

Copyright 2021-2023 Viam Inc. <br>
Apache 2.0
