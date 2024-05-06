import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from thop import profile
import time
import json
from ptflops import get_model_complexity_info

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        load_key = self.opt['path'].get('param_key_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), load_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            for p in self.net_g_ema.parameters():
                p.requires_grad = False
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.log_dict['l_g_pix'] = 0
        else:
            self.cri_pix = None

        if train_opt.get('artifacts_opt'):
            self.cri_artifacts = build_loss(train_opt['artifacts_opt']).to(self.device)
            self.log_dict['l_g_artifacts'] = 0
        else:
            self.cri_artifacts = None
        
        if train_opt.get('finetune_opt'):
            self.cri_finetune = build_loss(train_opt['finetune_opt'] | {'scale': self.opt['scale']}).to(self.device)
            self.log_dict['l_g_finetune'] = 0
        else:
            self.cri_finetune = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            self.log_dict['l_g_percep'] = 0
            if self.cri_perceptual.style_weight > 0:
                self.log_dict['l_g_style'] = 0
        else:
            self.cri_perceptual = None

        self.with_metrics = self.opt.get('metrics') is not None
        if self.with_metrics:
            self.metric_results = {f'm_{metric}': 0 for metric in self.opt['metrics'].keys()}
            self.log_dict |= self.metric_results

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        print(f'Using ema:{hasattr(self, "net_g_ema")}')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        # self.cycle_degredation()

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_g_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_g_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_g_style'] = l_style
        if self.cri_finetune:
            l_ft = self.cri_finetune(self.output, self.output)
            loss_dict['l_g_finetune'] = l_ft
            l_total += l_ft

        l_total.backward()
        self.optimizer_g.step()

        loss_dict |= self.calculate_metrics_on_iter()
        self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema') and self.opt.get('use_ema', False):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt.get('metrics') is not None
        eval_FID = self.opt.get('FID') is not None
        if with_metrics:
            self.metric_results_val_images = {metric: dict() for metric in self.opt['metrics'].keys()}
            self.metric_results_val = {metric: 0 for metric in self.opt['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')
        if eval_FID:
            FID_dataloader = []
        
        save_path = osp.join(self.opt['path']['visualization'], str(current_iter)) if self.opt['is_train'] else osp.join(self.opt['path']['visualization'], dataset_name)
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            if eval_FID:
                FID_dataloader.append((self.output.clone().cpu(), val_data['gt']))

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['metrics'].items():
                    metric_data = dict(img1=self.output, img2=self.gt)
                    self.metric_results_val_images[name][f'{img_name}.png'] = calculate_metric(metric_data, opt_).item()

            if save_img:
                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['result']])
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']])
                    del self.gt

                # tentative for out of GPU memory
                del self.lq
                del self.output
                torch.cuda.empty_cache()

                if save_img:
                    # if self.opt['is_train']:
                    save_img_path = osp.join(save_path, f'{img_name}.png')
                    # else:
                        # if self.opt['val']['suffix']:
                        #     save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                        #                             f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        # else:
                        # save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                        #                         f'{img_name}.png')
                    imwrite(sr_img, save_img_path)

            
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results_val.keys():
                self.metric_results_val[metric] = sum(self.metric_results_val_images[metric].values()) / (idx + 1)
            
        if self.opt.get('FID') is not None:
            metric_data = dict(data_generator = FID_dataloader)
            self.metric_results_val_images['FID'] = calculate_metric(metric_data, self.opt['FID']).item()
            self.metric_results_val['FID'] = self.metric_results_val_images['FID']
        

        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        # saving file with metrics
        if save_img:
            with open(osp.join(save_path, self.opt['path'].get('metric_names', 'metrics') + '.json'), 'w') as f:
                json.dump(self.metric_results_val_images, f)
        del self.metric_results_val_images

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results_val.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results_val.items():
                tb_logger.add_scalar(f'val/metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
    
    def calculate_metrics_on_iter(self):
        if self.with_metrics:
            # calculate metrics
            # output = self.output_ema if hasattr(self, 'net_g_ema') else self.output
            output = self.output
            for name, opt_ in self.opt['metrics'].items():
                metric_data = dict(img1=output, img2=self.gt)
                self.metric_results[f'm_{name}'] = calculate_metric(metric_data, opt_)
            return self.metric_results
        return dict()

    def get_samples_visualise(self, imdict):
        network = self.net_g_ema if hasattr(self, 'net_g_ema') and self.opt['use_ema'] else self.net_g
        device = torch.device('cuda' if self.opt['num_gpu'] != 0 else 'cpu')
        self.lq = torch.cat((imdict['tr_images'], imdict['tt_images']), dim=0).to(device)
        masks = torch.cat((imdict['tr_masks'], imdict['tt_masks']), dim=0).to(device)
        network.eval()
        with torch.no_grad():
            self.output = network(self.lq)
        network.train()
        del self.lq
        return {'logits': self.output, 'masks': masks, 'metrics': self.opt['metrics']}
