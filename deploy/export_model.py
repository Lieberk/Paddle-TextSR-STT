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

import os
import paddle
from model import tbsrn
paddle.set_device("cpu")
import warnings
warnings.filterwarnings("ignore")


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Training', add_help=add_help)
    parser.add_argument('--arch', default='tbsrn', choices=['tbsrn'])
    parser.add_argument('--STN', default=True, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--scale_factor', type=int, default=2, help='')
    parser.add_argument('--img_width', default=128, help='image width to export')
    parser.add_argument('--img_height', default=32, help='image height to export')
    parser.add_argument('--pretrained', default='./checkpoint/tbsrn_crnn_train/model_best.pdparams', help='use pretrained values')
    parser.add_argument(
        '--save_inference_dir', default='deploy', help='path where to save')
    args = parser.parse_args()
    return args


def export(args):
    if args.arch == 'tbsrn':
        model = tbsrn.TBSRN(scale_factor=args.scale_factor, width=args.img_width, height=args.img_height,
                            STN=args.STN, mask=args.mask, srb_nums=args.srb, hidden_units=args.hd_u)
    else:
        raise ValueError
    model_dict = paddle.load("%s" % args.pretrained)['state_dict_G']
    model.set_dict(model_dict)

    shape = [1, 3, args.img_height, args.img_width]
    model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print(f"inference model has been saved into {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)
