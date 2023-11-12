from ultralytics.models import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr
from utils import create_unique_path
import numpy as np
import time
import torch
import torch.nn as nn
from torch import distributed as dist
import warnings
from torch.nn.utils import spectral_norm
from torchvision.ops import roi_align, box_convert


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
    def __init__(self, alpha=1.0):
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
        self.model = nn.Sequential(
            SpectralConv2d(dim_in, dim_h // 2, kernel_size=1),
            nn.Flatten(),
            nn.LazyLinear(dim_h),
            SpectralLinear(dim_h, dim_o),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, chs=None):
        super().__init__()
        if chs is None:
            chs = [64, 128, 256]
            self.chs = chs
            self.f_len = len(chs)
        self.p = nn.ModuleList([
            nn.Sequential(
                SpectralConv2d(chs[i] if i == 0 else chs[i] * 2, chs[i + 1] if i + 1 < len(chs) else chs[i],
                               kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ) for i in range(len(chs))
        ])
        self.head = DiscriminatorHead(chs[-1], 256)

    def forward(self, fs: list[torch.tensor]):
        assert len(fs) == self.f_len, f'Expected {self.f_len} feature maps, got {len(fs)}'
        x = self.p[0](fs[0])
        for i in range(1, len(fs)):
            x = torch.cat((x, fs[i]), dim=1)
            x = self.p[i](x)
        return self.head(x)


class ProjectionHeads(nn.Module):
    def __init__(self, n_heads, dim=512, depth=2, act=None):
        super().__init__()
        self.n_heads = n_heads
        if act is None:
            act = nn.Identity
        elif isinstance(act, str):
            act = getattr(nn, act)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(dim // 2),
                act(),
                *[nn.Sequential(
                    nn.LazyLinear(dim // 2),
                    act(),
                ) for _ in range(depth - 2)],
                nn.LazyLinear(dim),
                act(),
            ) for _ in range(n_heads)
        ])

    def forward(self, x, h_idx):
        return self.heads[h_idx](x)


class CustomTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.model_hook_handler = []
        self.model_hook_layer_idx: list[int] = [2, 4, 6]
        self.roi_size = list(reversed([20 * 2 ** i for i in range(len(self.model_hook_layer_idx))]))
        self.model_hooked_features: None | list[torch.tensor] = None
        self.projection_model = ProjectionHeads(n_heads=len(self.model_hook_layer_idx))
        self.discriminator_model = Discriminator()

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
                    if self.model_hooked_features is not None:
                        bbox_batch_idx = batch['batch_idx'].unsqueeze(-1)
                        bbox = batch['bboxes']
                        bbox = box_convert(bbox, 'cxcywh', 'xyxy')
                        for fidx, f in enumerate(self.model_hooked_features):
                            f_bbox = bbox * f.shape[-1]
                            f_bbox = torch.cat((bbox_batch_idx, f_bbox), dim=-1)
                            f = roi_align(f, f_bbox, output_size=self.roi_size[fidx], aligned=True)
                            print()

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
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
                    self.save_model()
                    self.run_callbacks('on_model_save')

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
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    model.train(CustomTrainer, data='data/ACDC-fog.yaml', epochs=1, workers=3)


def plot_discriminator(path='runs/plot/discriminator'):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(create_unique_path(path))
    model = Discriminator()
    br = 30
    dummy_input = [torch.rand((1, model.chs[i], br // (2 ** i), br // (2 ** i))) for i in range(len(model.chs))]
    model(dummy_input)
    writer.add_graph(model, input_to_model=[dummy_input])
    writer.close()


if __name__ == '__main__':
    main()
    # plot_discriminator()
