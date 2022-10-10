import os
import yaml
import argparse
from easydict import EasyDict
import copy
import paddle
import logging
from interfaces import base
from utils.util import str_filt


class TextSR(base.TextBase):
    def train(self):
        self.config.TRAIN.epochs = args.epochs
        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']

        aster, aster_info = self.CRNN_init()
        optimizer_G = self.optimizer_init(model)

        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []

        for epoch in range(cfg.epochs):
            for j, data in (enumerate(train_loader)):
                model.train()
                for p in model.parameters():
                    p.stop_gradient = False
                iters = len(train_loader) * epoch + j

                images_hr, images_lr, label_strs = data
                sr_img = model(images_lr)
                loss, mse_loss, attention_loss, recognition_loss = image_crit(sr_img, images_hr, label_strs)

                self.writer.add_scalar('loss/mse_loss', mse_loss.item())
                self.writer.add_scalar('loss/position_loss', attention_loss.item())
                self.writer.add_scalar('loss/content_loss', recognition_loss.item())

                loss_im = loss * 100

                optimizer_G.clear_grad()
                loss_im.backward()
                optimizer_G.step()

                if iters % cfg.displayInterval == 0:
                    logging.info('Epoch: [{}][{}/{}]\t'
                                 'total_loss {:.3f} \t'
                                 'mse_loss {:.3f} \t'
                                 'attention_loss {:.3f} \t'
                                 'recognition_loss {:.3f} \t'
                                 .format(epoch, j + 1, len(train_loader),
                                         # self.vis_dir,
                                         loss_im.item(),
                                         mse_loss.item(),
                                         attention_loss.item(),
                                         recognition_loss.item()
                                         ))

                if iters % cfg.VAL.valInterval == 0:
                    logging.info('======================================================')
                    current_acc_dict = {}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        logging.info('evaling %s' % data_name)
                        metrics_dict = self.eval(model, val_loader, image_crit, iters, aster, aster_info, data_name)
                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:

                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            logging.info('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            logging.info('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        logging.info('saving best model')
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True,
                                             converge_list, self.args.exp_name)

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, False, converge_list,
                                         self.args.exp_name)

    def get_crnn_pred(self, outputs):
        alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'
        predict_result = []
        for output in outputs:
            max_index = paddle.topk(output, k=1, axis=1)[1]
            out_str = ""
            last = ""
            for i in max_index:
                if alphabet[i] != last:
                    if i != 0:
                        out_str += alphabet[i]
                        last = alphabet[i]
                    else:
                        last = ""
            predict_result.append(out_str)
        return predict_result

    def eval(self, model, val_loader, image_crit, index, recognizer, aster_info, mode):
        for p in model.parameters():
            p.stop_gradient = True
        for p in recognizer.parameters():
            p.stop_gradient = True
        model.eval()
        recognizer.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0,
                       'images_and_labels': []}
        for i, data in (enumerate(val_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            images_sr = model(images_lr)

            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            recognizer_dict_sr = self.parse_crnn_data(images_sr[:, :3, :, :])
            recognizer_output_sr = recognizer(recognizer_dict_sr)
            outputs_sr = recognizer_output_sr.transpose([1, 0, 2])
            predict_result_sr = self.get_crnn_pred(outputs_sr)
            metric_dict['images_and_labels'].append(
                (images_hr.cpu(), images_sr.cpu(), label_strs, predict_result_sr))

            cnt = 0
            for pred, target in zip(predict_result_sr, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct += 1
                cnt += 1

            sum_images += val_batch_size
            paddle.device.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        logging.info('PSNR {:.2f} | SSIM {:.4f}\t'
                     .format(float(psnr_avg), float(ssim_avg), ))
        accuracy = round(n_correct / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        logging.info('sr_accuray: %.2f%%' % (accuracy * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg

        if mode == 'easy':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy)
        if mode == 'medium':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy)
        if mode == 'hard':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy)

        return metric_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tbsrn', choices=['tbsrn'])
    parser.add_argument('--text_focus', default=True)
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs to train the model')
    parser.add_argument('--output_dir', default='./checkpoint/', help='output dir')
    parser.add_argument('--exp_name', default='tbsrn_crnn_train', help='Type your experiment name')
    parser.add_argument('--test_data_dir', type=str, default='./dataset/mydata/test')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--resume', type=str, default='', help='')
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
    Mission = TextSR(config, args)
    Mission.train()
