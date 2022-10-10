import os
import sys
import paddle
import shutil
import string
import logging
import paddle.optimizer as optim

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

from model import tbsrn, crnn
import dataset.dataset as dataset
from dataset import lmdbDataset, alignCollate_real, lmdbDataset_real, alignCollate_syn, lmdbDataset_mix
from loss import text_focus_loss
from utils import utils_crnn, ssim_psnr
from utils.labelmaps import get_vocabulary
import numpy as np
from PIL import Image
from paddle.vision import transforms


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    return total_num


class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale
        self.output_dir = args.output_dir
        if self.args.syn:
            self.align_collate = alignCollate_syn
            self.load_dataset = lmdbDataset
        elif self.args.mixed:
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_mix
        else:
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_real
        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation
        }
        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TRAIN.max_len
        # self.vis_dir = self.args.vis_dir if self.args.vis_dir is not None else self.config.TRAIN.VAL.vis_dir
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.mask = self.args.mask
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)
        if self.config.TRAIN.train_only:
            self.clean_old_ckpt()
        self.logging = logging
        self.make_logger()
        self.make_writer()

    def make_logger(self):
        self.logging.basicConfig(filename="{}/{}/log.txt".format(self.output_dir, self.args.exp_name),
                                 level=self.logging.INFO,
                                 format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        self.logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.logging.info(str(self.args))

    def clean_old_ckpt(self):
        if os.path.exists('{}/{}'.format(self.output_dir, self.args.exp_name)):
            shutil.rmtree('{}/{}'.format(self.output_dir, self.args.exp_name))
            print(f'Clean the old checkpoint {self.args.exp_name}')
        os.mkdir('{}/{}'.format(self.output_dir, self.args.exp_name))

    def make_writer(self):
        self.writer = tb.SummaryWriter('{}/{}'.format(self.output_dir, self.args.exp_name))

    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(
                    self.load_dataset(root=data_dir_,
                                      voc_type=cfg.voc_type,
                                      max_len=cfg.max_len))
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        train_loader = paddle.io.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=True)
        return train_dataset, train_loader

    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TRAIN
        test_dataset = self.load_dataset(root=dir_,
                                         voc_type=cfg.voc_type,
                                         max_len=cfg.max_len,
                                         test=True,
                                         )
        test_loader = paddle.io.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=False)
        return test_dataset, test_loader

    def generator_init(self):
        cfg = self.config.TRAIN
        if self.args.arch == 'tbsrn':
            model = tbsrn.TBSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = text_focus_loss.TextFocusLoss(self.args)
        else:
            raise ValueError
        if self.args.arch != 'bicubic':
            if self.resume is not '':
                self.logging.info('loading pre-trained model from %s ' % self.resume)
                if self.config.TRAIN.ngpu == 1:
                    model.load_dict(paddle.load(self.resume)['state_dict_G'])
                else:
                    model.load_dict(
                        {'module.' + k: v for k, v in paddle.load(self.resume)['state_dict_G'].items()})

        para_num = get_parameter_number(model)
        self.logging.info('Total Parameters {}'.format(para_num))

        return {'model': model, 'crit': image_crit}

    def optimizer_init(self, model):
        cfg = self.config.TRAIN
        clip = paddle.nn.ClipGradByNorm(clip_norm=0.25)
        optimizer = optim.Adam(learning_rate=cfg.lr,
                               parameters=model.parameters(),
                               beta1=cfg.beta1,
                               beta2=0.999,
                               grad_clip=clip)
        return optimizer

    def save_checkpoint(self, netG, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list, exp_name):
        # ckpt_path = os.path.join('checkpoint', exp_name, self.vis_dir)
        ckpt_path = os.path.join('{}'.format(self.output_dir), exp_name)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        save_dict = {
            'state_dict_G': netG.state_dict(),
            'info': {'arch': self.args.arch, 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                     'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
            'best_history_res': best_acc_dict,
            'best_model_info': best_model_info,
            'converge': converge_list
        }
        if is_best:
            paddle.save(save_dict, os.path.join(ckpt_path, 'model_best.pdparams'))
        else:
            paddle.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pdparams'))

    def CRNN_init(self):
        model = crnn.CRNN(32, 1, 37, 256)
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        model_path = self.config.TRAIN.VAL.crnn_pretrained
        self.logging.info('loading pretrained crnn model from %s' % model_path)
        model.load_dict(paddle.load(model_path))
        return model, aster_info

    def parse_crnn_data(self, imgs_input):
        imgs_input = paddle.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def transform_(self, path):
        img = Image.open(path)
        img = img.resize((64, 16), Image.BICUBIC)
        img_tensor = transforms.ToTensor()(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = transforms.ToTensor()(mask)
            img_tensor = paddle.concat((img_tensor, mask), 0)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)

