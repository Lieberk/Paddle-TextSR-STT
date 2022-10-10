import yaml
import argparse
from easydict import EasyDict
import os
import time
import paddle
import logging
from tqdm import tqdm
from utils.util import str_filt
from interfaces import base


class TextSR_TETS(base.TextBase):
    def test(self):
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        items = os.listdir(self.test_data_dir)
        for test_dir in items:
            test_data, test_loader = self.get_test_data(os.path.join(self.test_data_dir, test_dir))
            logging.info('evaling %s' % test_dir)
            if self.args.rec == 'crnn':
                crnn, _ = self.CRNN_init()
                crnn.eval()
            else:
                raise ValueError
            if self.args.arch != 'bicubic':
                for p in model.parameters():
                    p.stop_gradient = True
                model.eval()
            n_correct = 0
            sum_images = 0
            metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
            current_acc_dict = {test_dir: 0}
            time_begin = time.time()
            sr_time = 0
            with tqdm(unit='it', total=len(test_loader)) as pbar:
                for i, data in (enumerate(test_loader)):
                    images_hr, images_lr, label_strs = data
                    val_batch_size = images_lr.shape[0]
                    sr_beigin = time.time()
                    images_sr = model(images_lr)

                    sr_end = time.time()
                    sr_time += sr_end - sr_beigin
                    metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
                    metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

                    if self.args.rec == 'crnn':
                        crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                        crnn_output = crnn(crnn_input)
                        _, preds = crnn_output.topk(k=1, axis=2)
                        preds = preds.transpose([1, 0, 2]).reshape([-1])
                        preds_size = paddle.to_tensor([crnn_output.shape[0]] * val_batch_size, dtype='int32')
                        pred_str_sr = self.converter_crnn.decode(preds, preds_size, raw=False)
                    else:
                        raise ValueError
                    for pred, target in zip(pred_str_sr, label_strs):
                        if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                            n_correct += 1
                    sum_images += val_batch_size
                    paddle.device.cuda.empty_cache()
                    pbar.update()
            time_end = time.time()
            psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
            ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
            acc = round(n_correct / sum_images, 4)
            fps = sum_images / (time_end - time_begin)
            psnr_avg = round(psnr_avg.item(), 6)
            ssim_avg = round(ssim_avg.item(), 6)
            current_acc_dict[test_dir] = float(acc)
            result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
            logging.info(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tbsrn', choices=['tbsrn'])
    parser.add_argument('--text_focus', default=True)
    parser.add_argument('--exp_name', default='tbsrn_crnn_eval', help='Type your experiment name')
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
    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    Mission = TextSR_TETS(config, args)
    Mission.test()
