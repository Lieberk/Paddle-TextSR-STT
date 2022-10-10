# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import paddle
from paddle import inference
import numpy as np
from PIL import Image
from paddle.vision import transforms
from model import crnn
from utils import utils_crnn
import string
from utils.util import save_image


class InferenceEngine(object):
    """InferenceEngine

    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensors, self.output_tensors = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

        # build transforms
        self.transforms = transforms.ToTensor()

        self.mask = self.args.mask
        if self.args.rec == 'crnn':
            self.crnn_model = crnn.CRNN(32, 1, 37, 256)
            model_path = self.args.crnn_pretrained
            self.crnn_model.load_dict(paddle.load(model_path))
            self.crnn_model.eval()
        else:
            raise ValueError
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)

        # wamrup
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                print(idx)
                x = np.random.rand(1, 3, self.args.img_width,
                                   self.args.img_height).astype("float32")
                self.input_tensors.copy_from_cpu(x)
                self.predictor.run()
                self.output_tensors.copy_to_cpu()
        return

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()
            # The thread num should not be greater than the number of cores in the CPU.
            config.set_cpu_math_library_num_threads(4)

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_tensors = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_tensors = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return predictor, config, input_tensors, output_tensors,

    def preprocess(self, img_path):
        """preprocess
        Preprocess to the input.
        Args:
            img_path: Image path.
        Returns: Input data after preprocess.
        """
        img = Image.open(img_path)
        img = img.resize((64, 16), Image.BICUBIC)
        img_tensor = self.transforms(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.transforms(mask)
            img_tensor = paddle.concat((img_tensor, mask), 0)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def postprocess(self, x):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            x: Inference engine output.
        Returns: Output data after predict.
        """
        imgs_input = paddle.nn.functional.interpolate(x, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        crnn_input = 0.299 * R + 0.587 * G + 0.114 * B
        crnn_output = self.crnn_model(crnn_input)
        _, preds = crnn_output.topk(k=1, axis=2)
        preds = preds.transpose([1, 0, 2]).reshape([-1])
        preds_size = paddle.to_tensor([crnn_output.shape[0]] * 1, dtype='int32')
        pred_str_sr = self.converter_crnn.decode(preds, preds_size, raw=False)
        return pred_str_sr

    def run(self, x):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        self.input_tensors[0].copy_from_cpu(x)
        self.predictor.run()
        output = [paddle.to_tensor(y.copy_to_cpu()) for y in self.output_tensors]
        return output


def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)
    parser.add_argument(
        "--model_dir", default="deploy", help="inference model dir")
    parser.add_argument('--rec', default='crnn', choices=['crnn'])
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--crnn_pretrained', type=str, default='./dataset/mydata/crnn.pdparams', help='')
    parser.add_argument(
        "--use_gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument('--img_width', default=256, help='image width to export')
    parser.add_argument('--img_height', default=64, help='image height to export')
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--image_name', type=str, default='demo1.png')
    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")
    parser.add_argument(
        "--save_inference_dir", default="deploy", help="inference model dir")
    args = parser.parse_args()
    return args


def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    global autolog
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="classification",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img_path = os.path.join(args.demo_dir, args.image_name)
    img = inference_engine.preprocess(img_path)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(img.cpu().numpy())[0]
    save_image(output[:, :3, :, :], os.path.join(args.demo_dir, 'sr_' + args.image_name), padding=0)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    pred_str_lr = inference_engine.postprocess(img)
    pred_str_sr = inference_engine.postprocess(output)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print('{} ===> {}'.format(pred_str_lr, pred_str_sr))


if __name__ == "__main__":
    args = get_args()
    infer_main(args)
