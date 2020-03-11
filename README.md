# Tensorrt Inference Server Client

**Table Of Contents**
- [Description](#description)
- [Resource](#Resource)
- [Usage](#Usage)
- [License](#License)

## Description
This demo implements request trtis to inference.

## Resource
- [Docker image](https://pan.baidu.com/s/18OJS9EW8W5Z1O1_Uc27ZAQ), password: 540o

## Usage
- run the tensorrt docker image
	- `docker run -it --rm --net=host -v$PWD:/workspace/trt tensorrtserver_client`
- cd /workspace/trt/python
	- `python ak_image_client.py -m yolov3_608_trt -u X.X.X.X:XXX /workspace/trt/image/mayday.jpg`

## License
For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.
