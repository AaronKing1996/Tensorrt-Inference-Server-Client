#!/usr/bin/python

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
from builtins import range
from PIL import Image, ImageDraw
from tensorrtserver.api import *
import tensorrtserver.api.model_config_pb2 as model_config
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

FLAGS = None

def model_dtype_to_np(model_dtype):
    if model_dtype == model_config.TYPE_BOOL:
        return np.bool
    elif model_dtype == model_config.TYPE_INT8:
        return np.int8
    elif model_dtype == model_config.TYPE_INT16:
        return np.int16
    elif model_dtype == model_config.TYPE_INT32:
        return np.int32
    elif model_dtype == model_config.TYPE_INT64:
        return np.int64
    elif model_dtype == model_config.TYPE_UINT8:
        return np.uint8
    elif model_dtype == model_config.TYPE_UINT16:
        return np.uint16
    elif model_dtype == model_config.TYPE_FP16:
        return np.float16
    elif model_dtype == model_config.TYPE_FP32:
        return np.float32
    elif model_dtype == model_config.TYPE_FP64:
        return np.float64
    elif model_dtype == model_config.TYPE_STRING:
        return np.dtype(object)
    return None

def parse_model(url, protocol, model_name, batch_size, verbose=False):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    ctx = ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()

    if model_name not in server_status.model_status:
        raise Exception("unable to get status for '" + model_name + "'")

    status = server_status.model_status[model_name]
    config = status.config

    input = config.input[0]
    output = config.output

    # Model specifying maximum batch size of 0 indicates that batching
    # is not supported and so the input tensors do not expect an "N"
    # dimension (and 'batch_size' should be 1 so that only a single
    # image instance is inferred at a time).
    max_batch_size = config.max_batch_size
    if max_batch_size == 0:
        if batch_size != 1:
            raise Exception("batching not supported for model '" + model_name + "'")
    else: # max_batch_size > 0
        if batch_size > max_batch_size:
            raise Exception("expecting batch size <= {} for model {}".format(max_batch_size, model_name))

    # Model input must have 3 dims, either CHW or HWC
    if len(input.dims) != 3:
        raise Exception(
            "expecting input to have 3 dimensions, model '{}' input has {}".format(
                model_name, len(input.dims)))

    # Variable-size dimensions are not currently supported.
    for dim in input.dims:
        if dim == -1:
            raise Exception("variable-size dimension in model input not supported")

    if ((input.format != model_config.ModelInput.FORMAT_NCHW) and
        (input.format != model_config.ModelInput.FORMAT_NHWC)):
        raise Exception("unexpected input format " + model_config.ModelInput.Format.Name(input.format) +
                        ", expecting " +
                        model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NCHW) +
                        " or " +
                        model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NHWC))

    if input.format == model_config.ModelInput.FORMAT_NHWC:
        h = input.dims[0]
        w = input.dims[1]
        c = input.dims[2]
    else:
        c = input.dims[0]
        h = input.dims[1]
        w = input.dims[2]

    return (input.name, output, c, h, w, input.format, model_dtype_to_np(input.data_type))

def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """在原始输入图片上标记bounding box没然后返回结果.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """

    draw = ImageDraw.Draw(image_raw)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-a', '--async', action="store_true", required=False, default=False,
                        help='Use asynchronous inference API')
    parser.add_argument('--streaming', action="store_true", required=False, default=False,
                        help='Use streaming inference API. ' +
                        'The flag is only available with gRPC protocol.')
    parser.add_argument('-m', '--model-name', type=str, required=True,
                        help='Name of model')
    parser.add_argument('-x', '--model-version', type=int, required=False,
                        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b', '--batch-size', type=int, required=False, default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-c', '--classes', type=int, required=False, default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument('-s', '--scaling', type=str, choices=['NONE', 'INCEPTION', 'VGG'],
                        required=False, default='NONE',
                        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='HTTP',
                        help='Protocol (HTTP/gRPC) used to ' +
                        'communicate with inference service. Default is HTTP.')
    parser.add_argument('image_filename', type=str, nargs='?', default=None,
                        help='Input image / Input folder.')
    parser.add_argument('-o', '--output', type=str, required=False, default='../output/',
                        help='Output path')
    FLAGS = parser.parse_args()

    protocol = ProtocolType.from_str(FLAGS.protocol)

    if FLAGS.streaming and protocol != ProtocolType.GRPC:
        raise Exception("Streaming is only allowed with gRPC protocol")

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    input_name, outputs, c, h, w, format, dtype = parse_model(
        FLAGS.url, protocol, FLAGS.model_name,
        FLAGS.batch_size, FLAGS.verbose)

    ctx = InferContext(FLAGS.url, protocol, FLAGS.model_name,
                       FLAGS.model_version, FLAGS.verbose, 0, FLAGS.streaming)

    filenames = []
    if os.path.isdir(FLAGS.image_filename):
        filenames = [os.path.join(FLAGS.image_filename, f)
                     for f in os.listdir(FLAGS.image_filename)
                     if os.path.isfile(os.path.join(FLAGS.image_filename, f))]
    else:
        filenames = [FLAGS.image_filename]

    filenames.sort()

    # Preprocess the images into input data according to model

    # yolov3网络的输入size，HW顺序
    input_resolution_yolov3_HW = (608, 608)
    # 创建一个预处理来处理任意图片，以符合yolov3的输入
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    shape_orig_WH = []

    # requirements
    image_data = []
    image_raws = []

    for filename in filenames:
        image_raw, image = preprocessor.process(filename)
        image_data.append(image[0])
        image_raws.append(image_raw)
        shape_orig_WH.append(image_raw.size)

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    results = []
    result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False
    ak_result = {}

    output_name = dict()
    for output in outputs:
        output_name[output.name] = InferContext.ResultFormat.RAW

    while not last_request:
        input_filenames = []
        input_batch = []
        for idx in range(FLAGS.batch_size):
            input_filenames.append(filenames[image_idx])
            input_batch.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True

        result_filenames.append(input_filenames)

        # Send request
        # if not FLAGS.async:
        #     results.append(ctx.run(
        #         { input_name : input_batch },
        #         { output_name : (InferContext.ResultFormat.CLASS, FLAGS.classes) },
        #         batch_size=FLAGS.batch_size))
        # else:
        #     request_ids.append(ctx.async_run(
        #         { input_name : input_batch },
        #         { output_name : (InferContext.ResultFormat.CLASS, FLAGS.classes) },
        #         batch_size=FLAGS.batch_size))
        results.append(ctx.run(
                {input_name: input_batch},
                output_name,
                batch_size=FLAGS.batch_size))

    # For async, retrieve results according to the send order
    # if FLAGS.async:
    #     for request_id in request_ids:
    #         results.append(ctx.get_async_run_results(request_id, True))

    # yolov3输出的三个map的shapeOutput shapes expected by the post-processor
    output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]

    # postprocess(results[idx], result_filenames[idx], FLAGS.batch_size)
    # '''后处理'''
    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,  # 对象覆盖的阈值，[0,1]之间
                          "nms_threshold": 0.5,  # nms的阈值，[0,1]之间
                          "yolo_input_resolution": input_resolution_yolov3_HW}
    # 创建后处理类的实例
    postprocessor = PostprocessYOLO(**postprocessor_args)

    print("saving...")
    for idx in range(len(results)):
        trt_results = [results[idx]["082_convolutional"][0], results[idx]["094_convolutional"][0], results[idx]["106_convolutional"][0]]

        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_results, output_shapes)]
        # 运行后处理算法，并得到检测到对象的bounding box
        boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH[idx]))

        obj_detected_img = draw_bboxes(image_raws[idx], boxes, scores, classes, ALL_CATEGORIES)
        output_image_path = FLAGS.output + "{0}".format(filenames[idx].split("/")[-1])
        print(output_image_path)
        obj_detected_img.save(output_image_path)