# near-ir-person

This is a standalone [ml service module](https://app.viam.com/module/bijan/near-ir-person-detector) that only detects people seen by a near infrared camera. The underlying model is an SSD mobilenet V2 architecture exported to an ONNX format. The bundles the ONNX runtime within it. 

Configure this mlmodel service as a [modular resource](https://docs.viam.com/modular-resources/) and link it with the built-in `mlmodel` vision service to detect people when using a near infrared camera.

## Getting started

The first step is to configure a camera on your robot. [Here](https://docs.viam.com/components/camera/webcam/) is an example of how to configure a webcam. The next step is to configure both the mlmodel service and a vision service.

> [!NOTE]  
> Before configuring your mlmodel service or vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

## Configuration

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `mlmodel` type, then select the `near-ir-person` model. Enter a name for your service and click **Create**.

The ML Model service is configured fully upon creation, there are no attributes to fill out.

In order to see the detections from the model you will also need to create a vision service. Click on the **Services** subtab and click **Create service**. Select the `vision` type, then select the `mlmodel` model. Enter a name for your service and click **Create**. Then you should see the ml model service you configured in the drop down.

### Example Configuration

```json
{
  "modules": [
    {
      "type": "registry",
      "name": "viam_near-ir-person-detector",
      "module_id": "viam-labs:near-ir-person-detector",
      "version": "0.0.2"
    }
  ],
  "services": [
    {
      "name": "ir-person-mlmodel",
      "type": "mlmodel",
      "namespace": "rdk",
      "model": "viam-labs:mlmodel:near-ir-person",
      "attributes": {}
    },
    {
      "name": "ir-people",
      "type": "vision",
      "model": "mlmodel",
      "attributes": {
        "mlmodel_name": "ir-person-mlmodel"
      }
    }
  ]
}

```

> [!NOTE]  
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

### Attributes

There are no attributes associated with this module. It only runs the person detector for near IR cameras.

### Usage

This module is made for use with the following methods of the [mlmodel service API](https://docs.viam.com/services/ml/deploy/#api): 
- [`Infer()`](https://docs.viam.com/ml/deploy/#infer)
- [`Metadata()`](https://docs.viam.com/ml/deploy/#metadata)

This module expects one input tensor of the form `[1, 300, 300, 3]` i.e. a single RGB 300 x 300 resolution image.

## Visualize 

Once the `viam-labs:mlmodel:near-ir-person` and `viam:vision:mlmodel` services are in use, configure a [transform camera](https://docs.viam.com/components/camera/transform/) detections appear in your robot's field of vision.

