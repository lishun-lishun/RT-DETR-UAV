'''
by lyuwenyu
'''
import time 
import json
import datetime
import math

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from .training_plot import plot_training_curves, require_plot_backend


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        args = self.cfg 

        expected_world_size = args.yaml_cfg.get('expected_world_size')
        if expected_world_size is not None:
            world_size = dist.get_world_size()
            if world_size != expected_world_size:
                raise RuntimeError(
                    f'This config requires {expected_world_size} distributed processes, '
                    f'but got {world_size}. Launch it with torchrun '
                    f'--nproc_per_node={expected_world_size}.')

        plot_curves = bool(args.yaml_cfg.get('plot_training_curves', False))
        if plot_curves:
            require_plot_backend()

        self.train()

        if not isinstance(args.checkpoint_step, int) or args.checkpoint_step < 1:
            raise ValueError('checkpoint_step must be a positive integer')
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler,
                mert_config=args.yaml_cfg.get('MERT'))

            self.lr_scheduler.step()
            
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            improved = self._update_best_stat(test_stats, epoch)
            print('best_stat: ', self.best_stat)

            # Save AFTER validation so every checkpoint includes the updated
            # best AP. Only rank 0 writes, with one state snapshot per epoch.
            if self.output_dir and dist.is_main_process():
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                if improved:
                    checkpoint_paths.append(self.output_dir / 'best.pth')
                state = self.state_dict(epoch)
                state['validation_stats'] = test_stats
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(state, checkpoint_path)
                if improved:
                    print(f'Saved best.pth: epoch={epoch}, '
                          f'AP={self.best_stat["coco_eval_bbox"]:.6f}')


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'best_stat': dict(self.best_stat),
                        'world_size': dist.get_world_size(),
                        'batch_size_per_rank': self.train_dataloader.batch_size,
                        'global_batch_size': (dist.get_world_size() *
                                              self.train_dataloader.batch_size),
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if plot_curves:
                    plot_training_curves(
                        self.output_dir / 'log.txt',
                        self.output_dir / 'training_curves.png')

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def _update_best_stat(self, test_stats, epoch):
        """Choose the model by validation bbox AP@[.50:.95], not AP50/loss.

        A tie keeps the earlier best. Undefined COCO AP (-1), NaN and Inf
        cannot overwrite a valid best checkpoint.
        """
        scores = test_stats.get('coco_eval_bbox')
        if scores is None:
            return False
        try:
            ap = float(scores[0])
        except (IndexError, TypeError, ValueError):
            return False
        if not math.isfinite(ap) or not 0 <= ap <= 1:
            print(f'Skip best-model update: invalid validation bbox AP {ap}')
            return False
        if ap > self.best_stat.get('coco_eval_bbox', -math.inf):
            self.best_stat = {'epoch': epoch, 'coco_eval_bbox': ap}
            return True
        return False


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
