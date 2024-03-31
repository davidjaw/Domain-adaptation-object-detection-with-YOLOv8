from ultralytics.models import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr, emojis, clean_url
import ultralytics.utils.callbacks.tensorboard as tb_module
from utils import create_unique_path
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributed as dist
import warnings
from torch.nn.utils import spectral_norm
from torchvision.ops import roi_align, box_convert
from functools import partial
from copy import deepcopy
import random
import os


def seed_everything(seed=9527):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, alpha=1.):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class SpectralLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spectral_norm(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class SpectralConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        spectral_norm(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class DiscriminatorHead(nn.Module):
    def __init__(self, dim_in, dim_h, dim_o=1):
        super().__init__()
        self.to_flat = nn.Sequential(
            SpectralConv2d(dim_in, dim_h // 2, kernel_size=1),
            nn.Flatten(),
            nn.LazyLinear(dim_h),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.neck = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_h // 2, dim_h // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ) for _ in range(3)
        ])
        self.head = nn.Sequential(
            SpectralLinear(dim_h // 2 * 4, dim_h // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim_h // 2, dim_o, bias=False),
        )

    def forward(self, x):
        x = self.to_flat(x)
        x = x.split(x.shape[1] // 2, dim=1)
        xs = [x[0]]
        for m in self.neck:
            x = m(x[1]) if isinstance(x, tuple) else m(x)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        return self.head(x)


class Discriminator(nn.Module):
    def __init__(self, chs=None, amp=False):
        super().__init__()
        if chs is None:
            chs = [64, 128, 256]
            self.chs = chs
            self.f_len = len(chs)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.amp = amp
        self.p = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chs[i] if i == 0 else chs[i] * 2, 64, kernel_size=11, stride=2, padding=5, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, chs[i + 1] if i + 1 < len(chs) else chs[i], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(chs[i + 1] if i + 1 < len(chs) else chs[i]),
                nn.SiLU(inplace=True),
            ) for i in range(len(chs))
        ])
        self.head = DiscriminatorHead(chs[-1], 256)
        self.optim = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, fs: list[torch.tensor]):
        with torch.cuda.amp.autocast(self.amp):
            assert len(fs) == self.f_len, f'Expected {self.f_len} feature maps, got {len(fs)}'
            fs = [self.grl(f) for f in fs]
            x = self.p[0](fs[0])
            for i in range(1, len(fs)):
                x = torch.cat((x, fs[i]), dim=1)
                x = self.p[i](x)
            return self.head(x)


class CustomTrainer(DetectionTrainer):
    def __init__(self, target_domain_data_cfg, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        # Load target domain's dataset
        try:
            if target_domain_data_cfg in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                self.t_data = check_det_dataset(target_domain_data_cfg)
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(target_domain_data_cfg)}' error ❌ {e}")) from e

        self.t_trainset, self.t_testset = self.get_dataset(self.t_data)
        self.t_iter = None
        self.t_train_loader = None

        self.model_hook_handler = []
        self.model_hook_layer_idx: list[int] = [2, 4, 6]
        self.roi_size = list(reversed([20 * 2 ** i for i in range(len(self.model_hook_layer_idx))]))
        self.model_hooked_features: None | list[torch.tensor] = None
        self.discriminator_model = None
        self.projection_model = None
        self.additional_models = []
        self.add_callback('on_train_start', self.init_helper_model)

    def init_helper_model(self, *args, **kwargs):
        # self.amp will not be initialized from __init__, thus we need to initialize Dis here
        self.discriminator_model = Discriminator(amp=self.amp).to(self.device)
        self.additional_models.append(self.discriminator_model)
        # self.projection_model = ProjectionHeads(n_heads=len(self.model_hook_layer_idx))

    def get_t_batch(self):
        if self.t_iter is None:
            self.t_train_loader = self.get_dataloader(self.t_trainset, batch_size=self.batch_size, rank=RANK, mode='train')
            self.t_iter = iter(self.t_train_loader)
        try:
            batch = next(self.t_iter)
        except StopIteration:
            self.t_iter = iter(self.t_train_loader)
            batch = next(self.t_iter)
        return batch

    def activate_hook(self, layer_indices: list[int] = None):
        if layer_indices is not None:
            self.model_hook_layer_idx = layer_indices
        self.model_hooked_features = [None for _ in self.model_hook_layer_idx]
        self.model_hook_handler = \
            [self.model.model[l].register_forward_hook(self.hook_fn(i)) for i, l in enumerate(self.model_hook_layer_idx)]

    def deactivate_hook(self):
        if self.model_hook_handler is not None:
            for hook in self.model_hook_handler:
                hook.remove()
            self.model_hooked_features = None
            self.model_hook_handler = []

    def hook_fn(self, hook_idx):

        def hook(m, i, o):
            self.model_hooked_features[hook_idx] = o

        return hook

    def get_dis_output_from_hooked_features(self, batch):
        if self.model_hooked_features is not None:
            bbox_batch_idx = batch['batch_idx'].unsqueeze(-1)
            bbox = batch['bboxes']
            bbox = box_convert(bbox, 'cxcywh', 'xyxy')
            rois = []
            for fidx, f in enumerate(self.model_hooked_features):
                f_bbox = bbox * f.shape[-1]
                f_bbox = torch.cat((bbox_batch_idx, f_bbox), dim=-1)
                f_roi = roi_align(f, f_bbox.to(f.device), output_size=self.roi_size[fidx], aligned=True)
                rois.append(f_roi)
            dis_output = self.discriminator_model(rois)
            return dis_output
        else:
            return None

    def optimizer_step(self, optims: None | list[torch.optim.Optimizer] = None):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        if optims is not None:
            for o in optims:
                # check if the optimizer has gradients
                if o.param_groups[0]['params'][0].grad is not None:
                    self.scaler.unscale_(o)
        # Clip gradient norm
        max_norm = 10.0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)  # clip gradients
        if len(self.additional_models) > 0:
            for m in self.additional_models:
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=max_norm * 2)
        # Step optimizers
        self.scaler.step(self.optimizer)
        if optims is not None:
            for o in optims:
                # check if the optimizer has gradients
                if o.param_groups[0]['params'][0].grad is not None:
                    self.scaler.step(o)

        self.scaler.update()
        # Zero optimizers' grads
        self.optimizer.zero_grad()
        if optims is not None:
            for o in optims:
                o.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        self.activate_hook()
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.optimizer.zero_grad()
            source_critics, target_critics = None, None
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                    # Custom code here
                    # Get source domain features
                    source_critics = self.get_dis_output_from_hooked_features(batch)
                    # Get target domain features
                    t_batch = self.get_t_batch()
                    t_batch = self.preprocess_batch(t_batch)
                    t_loss, t_loss_item = self.model(t_batch)
                    target_critics = self.get_dis_output_from_hooked_features(t_batch)

                    self.loss += t_loss
                    if 6 < epoch < self.args.epochs - 50:
                        # Last 50 epochs are for OD only
                        threshold = 20
                        loss_d = (F.relu(torch.ones_like(source_critics) * threshold + source_critics)).mean()
                        loss_d += (F.relu(torch.ones_like(target_critics) * threshold - target_critics)).mean()
                    else:
                        loss_d = 0
                    self.loss += loss_d * 2
                # Backward
                self.scaler.scale(self.loss).backward()
                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step(optims=[self.discriminator_model.optim])
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    tb_module.WRITER.add_scalar('train/critic-source', source_critics.mean(), ni)
                    tb_module.WRITER.add_scalar('train/critic-target', target_critics.mean(), ni)
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):
                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.deactivate_hook()
                    self.save_model()
                    self.run_callbacks('on_model_save')
                    self.activate_hook()

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')

        self.deactivate_hook()
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')


def main():
    seed = 95 * 27
    kwargs = {
        'imgsz': 640,
        'epochs': 100,
        'val': False,
        'workers': 2,
        'batch': 32,
        'seed': seed,
    }
    seed_everything(seed)
    # RMD-Net
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    rmd_kwargs = deepcopy(kwargs)
    # rmd_kwargs['epochs'] = int(rmd_kwargs['epochs'] * 1.333)
    custom_trainer = partial(CustomTrainer, target_domain_data_cfg='data/ACDC-rain.yaml')
    model.train(custom_trainer, data='data/ACDC-fog.yaml', name='train_RMD', patience=0, **rmd_kwargs)
    model.val(data='data/ACDC-rain.yaml', name='val_RMD_rain')
    model.val(data='data/ACDC-fog.yaml', name='val_RMD_fog')

    # train on fog, test on rain
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    model.train(data='data/ACDC-fog.yaml', name='train_fog', **kwargs)
    model.val(data='data/ACDC-fog.yaml', name='val_fog_fog')
    model.val(data='data/ACDC-rain.yaml', name='val_fog_rain')

    # train on rain, test on rain
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    model.train(data='data/ACDC-rain.yaml', name='train_rain', **kwargs)
    model.val(data='data/ACDC-rain.yaml', name='val_rain_rain')
    model.val(data='data/ACDC-fog.yaml', name='val_rain_fog')

    # train on both rain and fog
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    model.train(data='data/ACDC-fog_rain.yaml', name='train_fog_rain-full-epoch', **kwargs)
    model.val(data='data/ACDC-rain.yaml', name='val_fograin_rain-full-epoch')
    model.val(data='data/ACDC-fog.yaml', name='val_fograin_fog-full-epoch')

    # train on both rain and fog
    kwargs['epochs'] //= 2
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    model.train(data='data/ACDC-fog_rain.yaml', name='train_fog_rain', **kwargs)
    model.val(data='data/ACDC-rain.yaml', name='val_fograin_rain')
    model.val(data='data/ACDC-fog.yaml', name='val_fograin_fog')


if __name__ == '__main__':
    main()
    # plot_discriminator()
