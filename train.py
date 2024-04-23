from ultralytics.models import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr, emojis, clean_url
import ultralytics.utils.callbacks.tensorboard as tb_module
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
import random
import os


def seed_everything(seed=9527):
    # 固定隨機種子, 藉此讓整體訓練過程可復現
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class GradientReversalFunction(torch.autograd.Function):
    # 梯度反轉函數，用於進行梯度反向傳播時改變梯度的方向
    @staticmethod
    def forward(ctx, x, alpha):
        """
        前向傳播，直接返回輸入 x。
        Args:
            ctx: 上下文物件，用於保存資訊以便反向傳播時使用
            x: 輸入的張量
            alpha: 梯度反轉層的係數，控制梯度反轉的強度
        Returns:
            與輸入 x 形狀相同的張量
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向傳播，將傳入的梯度乘以 -alpha，實現梯度反轉。
        Args:
            ctx: 保存的上下文資訊
            grad_output: 反向傳播到此處的梯度
        Returns:
            輸出的梯度乘以 -alpha, 並且對於 alpha 返回 None，因為 alpha 通常視為常數
        """
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, alpha=1.):
        """
        初始化梯度反轉層。
        Args:
            alpha: 梯度反轉的係數, 默認為 1.0
        """
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        模組的前向傳播邏輯, 透過使用 GradientReversalFunction 實現梯度反轉。
        Args:
            x: 輸入張量
        Returns:
            經過梯度反轉函數處理的張量
        """
        return GradientReversalFunction.apply(x, self.alpha)


class SpectralLinear(nn.Linear):
    # 使用頻譜歸一化的線性層
    def __init__(self, *args, **kwargs):
        """
        初始化頻譜歸一化的線性層，繼承自 nn.Linear，並對權重添加頻譜歸一化。
        """
        super().__init__(*args, **kwargs)
        spectral_norm(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class SpectralConv2d(nn.Conv2d):
    # 使用頻譜歸一化的二維卷積層
    def __init__(self, *args, **kwargs):
        """
        初始化頻譜歸一化的二維卷積層，繼承自 nn.Conv2d，並對權重添加頻譜歸一化。
        """
        super().__init__(*args, **kwargs)
        spectral_norm(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class DiscriminatorHead(nn.Module):
    def __init__(self, dim_in, dim_h, dim_o=1):
        """
        初始化鑒別器的頭部結構，包括多個層的序列。
        Args:
            dim_in: 輸入特徵的維度
            dim_h: 隱藏層的維度
            dim_o: 輸出層的維度，默認為 1
        """
        super().__init__()
        # 將輸入特徵展平 並通過一個線性層轉換到隱藏層維度
        self.to_flat = nn.Sequential(
            SpectralConv2d(dim_in, dim_h // 2, kernel_size=1),
            nn.Flatten(),
            nn.LazyLinear(dim_h),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 隱藏層的多個線性層
        self.neck = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_h // 2, dim_h // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ) for _ in range(3)
        ])
        # 最終的線性層
        self.head = nn.Sequential(
            SpectralLinear(dim_h // 2 * 4, dim_h // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim_h // 2, dim_o, bias=False),
        )

    def forward(self, x):
        # 通過一個線性層將輸入特徵展平
        x = self.to_flat(x)
        # 類似於 CSPNet 的結構，將展平後的特徵分成多份, 並最終聚合不同深度的特徵
        x = x.split(x.shape[1] // 2, dim=1)
        xs = [x[0]]
        for m in self.neck:
            x = m(x[1]) if isinstance(x, tuple) else m(x)
            xs.append(x)
        # 聚合不同深度的特徵
        x = torch.cat(xs, dim=1)
        return self.head(x)


class Discriminator(nn.Module):
    def __init__(self, chs=None, amp=False):
        """
        初始化鑒別器結構，包括梯度反轉層和多個卷積層。
        Args:
            chs: 通道數列表，定義了不同卷積層的輸入和輸出通道數
            amp: 是否使用自動混合精度訓練，默認為 False
        """
        super().__init__()
        # 此處是對應到 YOLO 架構中被 hook 的層的通道數
        if chs is None:
            chs = [64, 128, 256]
            self.chs = chs
            self.f_len = len(chs)
        # 初始化梯度反轉層
        self.grl = GradientReversalLayer(alpha=1.0)
        self.amp = amp
        # 用來提取不同深度特徵的卷積層, 每層的通道數由 chs 定義
        self.p = nn.ModuleList([
            nn.Sequential(
                # 第一層卷積層, 通過 11x11 的卷積核對特徵進行提取
                nn.Conv2d(chs[i] if i == 0 else chs[i] * 2, 64, kernel_size=11, stride=2, padding=5, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                # 第二層卷積層, 通過 1x1 的卷積核對特徵進行維度轉換
                nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                # 第三層卷積層, 通過 1x1 的卷積核對特徵進行維度轉換, 目標為次個 chs 級數的通道數
                nn.Conv2d(32, chs[i + 1] if i + 1 < len(chs) else chs[i], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(chs[i + 1] if i + 1 < len(chs) else chs[i]),
                nn.SiLU(inplace=True),
            ) for i in range(len(chs))
        ])
        # 聚合所有 chs 特徵後的頭部結構
        self.head = DiscriminatorHead(chs[-1], 256)
        # 與 detector model 分開的優化器, 僅用於訓練鑒別器本身
        self.optim = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, fs: list[torch.tensor]):
        with torch.cuda.amp.autocast(self.amp):
            assert len(fs) == self.f_len, f'Expected {self.f_len} feature maps, got {len(fs)}'
            # 在接到 detector model 的特徵後, 透過梯度反轉層對特徵進行梯度反轉
            # 即 detector model 將是 generator, 而鑒別器將會對 detector model 的特徵進行"域鑒別"
            fs = [self.grl(f) for f in fs]
            # 跨解析度的特徵聚合架構
            x = self.p[0](fs[0])
            for i in range(1, len(fs)):
                x = torch.cat((x, fs[i]), dim=1)
                x = self.p[i](x)
            # 最終透過頭部結構進行預測
            return self.head(x)


class CustomTrainer(DetectionTrainer):
    """
    此為自定義的訓練器, 透過此訓練器可以在訓練過程中同時進行 source domain 和 target domain 的訓練
    訓練迴圈是基於 ultralytics.models.yolo.detect.train.DetectionTrainer 進行擴充而來,
    整體的運行邏輯與 DetectionTrainer 類似, 但在訓練過程中會加入 domain adaptation 的相關邏輯與 discriminator 的訓練
    """
    def __init__(self, target_domain_data_cfg, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        # 額外載入 target domain 的資料集
        try:
            if target_domain_data_cfg in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                self.t_data = check_det_dataset(target_domain_data_cfg)
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(target_domain_data_cfg)}' error ❌ {e}")) from e
        # 初始化 target domain 的資料集與 dataloader
        self.t_trainset, self.t_testset = self.get_dataset(self.t_data)
        self.t_iter = None
        self.t_train_loader = None
        # 初始化 domain adaptation 相關的參數與 model hook 相關的參數
        self.model_hook_handler = []
        # 設定需要 hook 的層的索引, 這裡是 YOLO 的 backbone 中的 3 個層
        self.model_hook_layer_idx: list[int] = [2, 4, 6]
        # roi size 對應到不同層的特徵圖大小
        self.roi_size = list(reversed([20 * 2 ** i for i in range(len(self.model_hook_layer_idx))]))
        # 初始化 hook 的特徵
        self.model_hooked_features: None | list[torch.tensor] = None
        # 初始化 discriminator model 與 projection heads
        self.discriminator_model = None
        self.projection_model = None
        self.additional_models = []
        # 此處的 on_train_start 是在 DetectionTrainer 的訓練過程中的 callback, 用於初始化 helper model
        self.add_callback('on_train_start', self.init_helper_model)

    def init_helper_model(self, *args, **kwargs):
        # 由於 self.amp 變數是在 DetectionTrainer 的訓練過程中初始化的, 而 Discriminator 需要使用到 self.amp,
        # 因此要在整個 trainer 跟 model 被初始化結束並開始訓練前進行初始化, 才能使用相同的設定
        self.discriminator_model = Discriminator(amp=self.amp).to(self.device)
        # 將 discriminator model 加入到額外的模型列表中, 這樣在 optimizer_step 時可以一起進行優化
        self.additional_models.append(self.discriminator_model)

    def get_t_batch(self):
        # 獲取 target domain 的 batch
        if self.t_iter is None:
            # 由於每次被呼叫時僅取一個 batch, 因此將 dataloader 轉為 iterator 以便進行迭代
            self.t_train_loader = self.get_dataloader(self.t_trainset, batch_size=self.batch_size, rank=RANK, mode='train')
            self.t_iter = iter(self.t_train_loader)
        try:
            # Next 方法會在迭代結束時拋出 StopIteration 錯誤, 因此透過 try-except 來進行迭代
            batch = next(self.t_iter)
        except StopIteration:
            # 當迭代結束時, 重新初始化 dataloader 並再將其轉為 iterator
            self.t_iter = iter(self.t_train_loader)
            batch = next(self.t_iter)
        return batch

    def activate_hook(self, layer_indices: list[int] = None):
        # 啟動 hook, 透過 hook 可以獲取到指定層的特徵
        if layer_indices is not None:
            self.model_hook_layer_idx = layer_indices
        self.model_hooked_features = [None for _ in self.model_hook_layer_idx]
        self.model_hook_handler = \
            [self.model.model[l].register_forward_hook(self.hook_fn(i)) for i, l in enumerate(self.model_hook_layer_idx)]

    def deactivate_hook(self):
        # 停用 hook, 並移除所有的 hook
        if self.model_hook_handler is not None:
            for hook in self.model_hook_handler:
                hook.remove()
            self.model_hooked_features = None
            self.model_hook_handler = []

    def hook_fn(self, hook_idx):
        # 定義 hook 函數, 並將其儲存至指定的 list index 中
        def hook(m, i, o):
            self.model_hooked_features[hook_idx] = o
        return hook

    def get_dis_output_from_hooked_features(self, batch):
        # 將 hook 的特徵傳入 discriminator model 進行預測
        if self.model_hooked_features is not None:
            # 對資料維度和座標進行轉換
            bbox_batch_idx = batch['batch_idx'].unsqueeze(-1)
            bbox = batch['bboxes']
            bbox = box_convert(bbox, 'cxcywh', 'xyxy')
            # 將 bbox 進行縮放, 並透過 roi align 進行特徵提取
            rois = []
            for fidx, f in enumerate(self.model_hooked_features):
                f_bbox = bbox * f.shape[-1]
                f_bbox = torch.cat((bbox_batch_idx, f_bbox), dim=-1)
                f_roi = roi_align(f, f_bbox.to(f.device), output_size=self.roi_size[fidx], aligned=True)
                rois.append(f_roi)
            # 將不同解析度的特徵進行透過 discriminator model 進行特徵聚合與預測
            dis_output = self.discriminator_model(rois)
            return dis_output
        else:
            return None

    def optimizer_step(self, optims: None | list[torch.optim.Optimizer] = None):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        # 將 optimizer 的梯度進行放大以配合混合精度訓練
        self.scaler.unscale_(self.optimizer)
        # 若有傳入額外的 optimizer, 則也對其進行梯度放大 (此處應為 discriminator model 的 optimizer)
        if optims is not None:
            for o in optims:
                # check if the optimizer has gradients
                if o.param_groups[0]['params'][0].grad is not None:
                    self.scaler.unscale_(o)
        # 截斷過大的梯度
        max_norm = 10.0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)  # clip gradients
        if len(self.additional_models) > 0:
            for m in self.additional_models:
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=max_norm * 2)
        # 透過 scaler 進行 optimizer 的更新
        self.scaler.step(self.optimizer)
        if optims is not None:
            for o in optims:
                # check if the optimizer has gradients
                if o.param_groups[0]['params'][0].grad is not None:
                    self.scaler.step(o)
        # 更新 scaler
        self.scaler.update()
        # 重置 optimizer 梯度
        self.optimizer.zero_grad()
        if optims is not None:
            for o in optims:
                o.zero_grad()
        # 若有 EMA, 則進行 EMA 更新
        if self.ema:
            self.ema.update(self.model)

    def _do_train(self, world_size=1):
        # 調整分布式訓練的設定
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        # 訓練過程中的一些參數設定
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        # 此處呼叫了 on_train_start 的 callback, 對應到我們上面撰寫的初始化 discriminator model 的 callback
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        # 設定訓練過程中的參數
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        # 開始訓練迴圈, 先將模型的 hook 啟動
        self.activate_hook()
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # 更新 dataloader 的設定
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()
            # 分布式訓練的進度條
            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            # 初始化訓練過程中的參數, 歸零 loss, optimizer
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                # 呼叫 on_train_batch_start 的 callback (本章節沒有加入額外的 callback)
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
                # 前向傳播
                with torch.cuda.amp.autocast(self.amp):
                    # 先獲得 source domain 的 object detection loss 與 forward hook 的特徵
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    # 分布式訓練時, 將 loss 乘上 world_size 以避免分布式訓練時 loss 過小
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items
                    # 獲取 source domain 的 discriminator output
                    source_critics = self.get_dis_output_from_hooked_features(batch)
                    # 接下來換對 target domain 進行訓練
                    t_batch = self.get_t_batch()
                    # 對 target domain 進行 forward, 並獲取 target domain 的 discriminator output
                    t_batch = self.preprocess_batch(t_batch)
                    t_loss, t_loss_item = self.model(t_batch)
                    target_critics = self.get_dis_output_from_hooked_features(t_batch)
                    # 整合 source domain 與 target domain 的 object detection loss
                    self.loss += t_loss
                    # 前 6 個 epoch 與最後的 50 個 epoch 不進行 discriminator 的訓練,
                    # 前面不訓練是因為 object detection 尚未有能力獲得具代表性的特徵, 因此 domain adaptation 於此時較無意義
                    # 後面不訓練則是為了不讓 GAN 較不穩定的梯度影響 object detection 的表現
                    if 6 < epoch < self.args.epochs - 50:
                        # 計算 discriminator 的 loss, 採用 hinge loss
                        threshold = 20
                        loss_d = (F.relu(torch.ones_like(source_critics) * threshold + source_critics)).mean()
                        loss_d += (F.relu(torch.ones_like(target_critics) * threshold - target_critics)).mean()
                    else:
                        loss_d = 0
                    self.loss += loss_d * 2
                # 反向傳播
                self.scaler.scale(self.loss).backward()
                # 分布式訓練相關
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step(optims=[self.discriminator_model.optim])
                    last_opt_step = ni
                # 紀錄訓練過程中的一些參數, 並進行一些 log 的操作
                # 此處之後的 code 直到 deactive_hook 前都是 ultraalytics 的訓練過程中的 log 與訓練過程, 就不再進行額外註解
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
                    # 此處須注意, 由於 discriminator model 並不是 detector model 的一部分, 且儲存模型時不該儲存 hook 相關特徵
                    # 因此需要先停用 hook 再進行模型的儲存
                    self.deactivate_hook()
                    self.save_model()
                    self.run_callbacks('on_model_save')
                    # 由於接著要繼續訓練, 因此需要重新啟動 hook
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

        # 結束訓練後, 停用 hook
        self.deactivate_hook()
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')


def main():
    # 設定 Hyperparameters 和隨機種子
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
    # 先載入預訓練模型
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    # 初始化自定義的訓練器
    custom_trainer = partial(CustomTrainer, target_domain_data_cfg='data/ACDC-rain.yaml')
    # 透過自定義的訓練器進行訓練
    model.train(custom_trainer, data='data/ACDC-fog.yaml', name='train_RMD', patience=0, **kwargs)
    # 將訓練好的模型在 rain domain 上進行測試
    model.val(data='data/ACDC-rain.yaml', name='val_RMD_rain')
    # 將訓練好的模型在 fog domain 上進行測試
    model.val(data='data/ACDC-fog.yaml', name='val_RMD_fog')

    # 在 fog domain 上進行訓練, 一樣先載入預訓練模型
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    # 透過 ultryalitics 提供的預設訓練器進行訓練
    model.train(data='data/ACDC-fog.yaml', name='train_fog', **kwargs)
    # 將訓練好的模型在 fog domain 上進行測試
    model.val(data='data/ACDC-fog.yaml', name='val_fog_fog')
    # 將訓練好的模型在 rain domain 上進行測試
    model.val(data='data/ACDC-rain.yaml', name='val_fog_rain')

    # 在 rain domain 上進行訓練, 一樣先載入預訓練模型
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    # 透過 ultryalitics 提供的預設訓練器進行訓練
    model.train(data='data/ACDC-rain.yaml', name='train_rain', **kwargs)
    # 將訓練好的模型在 rain domain 上進行測試
    model.val(data='data/ACDC-rain.yaml', name='val_rain_rain')
    # 將訓練好的模型在 fog domain 上進行測試
    model.val(data='data/ACDC-fog.yaml', name='val_rain_fog')

    # 在 rain 和 fog domain 上進行訓練, 一樣先載入預訓練模型
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    # 透過 ultryalitics 提供的預設訓練器進行訓練
    model.train(data='data/ACDC-fog_rain.yaml', name='train_fog_rain-full-epoch', **kwargs)
    # 將訓練好的模型在 rain domain 上進行測試
    model.val(data='data/ACDC-rain.yaml', name='val_fograin_rain-full-epoch')
    # 將訓練好的模型在 fog domain 上進行測試
    model.val(data='data/ACDC-fog.yaml', name='val_fograin_fog-full-epoch')


if __name__ == '__main__':
    main()
