import os
import time
import yaml
import argparse
from easydict import EasyDict
import paddle
import logging
from interfaces import base
from utils.util import save_image


class TextSR_PREDICT(base.TextBase):
    def demo(self):
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'crnn':
            crnn, _ = self.CRNN_init()
            crnn.eval()
        else:
            raise ValueError
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.stop_gradient = True
            model.eval()

        time_begin = time.time()
        images_lr = self.transform_(os.path.join(self.args.demo_dir, self.args.image_name))
        images_sr = model(images_lr)
        save_image(images_sr[:, :3, :, :], os.path.join(self.args.demo_dir, 'sr_' + self.args.image_name), padding=0)

        if self.args.rec == 'crnn':
            crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
            crnn_output = crnn(crnn_input)
            _, preds = crnn_output.topk(k=1, axis=2)
            preds = preds.transpose([1, 0, 2]).reshape([-1])
            preds_size = paddle.to_tensor([crnn_output.shape[0]] * 1, dtype='int32')
            pred_str_sr = self.converter_crnn.decode(preds, preds_size, raw=False)

            crnn_input_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
            crnn_output_lr = crnn(crnn_input_lr)
            _, preds_lr = crnn_output_lr.topk(k=1, axis=2)
            preds_lr = preds_lr.transpose([1, 0, 2]).reshape([-1])
            preds_size = paddle.to_tensor([crnn_output_lr.shape[0]] * 1, dtype='int32')
            pred_str_lr = self.converter_crnn.decode(preds_lr, preds_size, raw=False)
        else:
            raise ValueError
        logging.info('{} ===> {}'.format(pred_str_lr, pred_str_sr))
        paddle.device.cuda.empty_cache()
        time_end = time.time()
        fps = time_end - time_begin
        logging.info('fps={}'.format(fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tbsrn', choices=['tbsrn'])
    parser.add_argument('--text_focus', default=True)
    parser.add_argument('--exp_name', default='tbsrn_crnn_predict', help='Type your experiment name')
    parser.add_argument('--test_data_dir', type=str, default='./dataset/mydata/test')
    parser.add_argument('--output_dir', default='./checkpoint/', help='output dir')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--resume', type=str, default='./checkpoint/tbsrn_crnn_train/model_best.pdparams', help='')
    parser.add_argument('--rec', default='crnn', choices=['crnn'])
    parser.add_argument('--STN', default=True, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--image_name', type=str, default='demo1.png')
    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    Mission = TextSR_PREDICT(config, args)
    Mission.demo()
