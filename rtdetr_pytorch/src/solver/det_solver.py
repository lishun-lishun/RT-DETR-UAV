'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    @staticmethod
    def _metric_value(value):
        """Return the primary scalar from a COCO metric list or a scalar."""
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    def _recover_best_stat(self):
        """Recover best bbox mAP from logs made by older checkpoints.

        Checkpoints created before best-model saving was added do not contain
        ``best_stat``. Reading log.txt makes resuming those runs safe and also
        lets us materialize the historical best checkpoint as best.pth.
        """
        log_path = self.output_dir / 'log.txt'
        best_stat = {'epoch': -1}
        if not log_path.exists():
            return best_stat

        with log_path.open('r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    score = self._metric_value(record['test_coco_eval_bbox'])
                    if score > best_stat.get('coco_eval_bbox', float('-inf')):
                        best_stat = {
                            'epoch': record['epoch'],
                            'coco_eval_bbox': score,
                        }
                except (json.JSONDecodeError, KeyError, TypeError, IndexError):
                    # Ignore an incomplete final line or unrelated old record.
                    continue

        return best_stat

    def _materialize_historical_best(self, best_stat):
        """Create best.pth from an already saved per-epoch checkpoint."""
        epoch = best_stat.get('epoch', -1)
        if epoch < 0 or not dist.is_main_process():
            return

        best_path = self.output_dir / 'best.pth'
        epoch_path = self.output_dir / f'checkpoint{epoch:04}.pth'
        if best_path.exists() or not epoch_path.exists():
            return

        state = torch.load(epoch_path, map_location='cpu')
        state['best_stat'] = best_stat
        torch.save(state, best_path)
        print(f'Recovered historical best checkpoint: {best_path}')
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # New checkpoints restore this value directly. For checkpoints produced
        # by older code, recover it from log.txt instead.
        best_stat = self.best_stat
        if best_stat.get('epoch', -1) < 0:
            best_stat = self._recover_best_stat()
            self.best_stat = best_stat
            if best_stat.get('epoch', -1) >= 0:
                print('Recovered best_stat from log.txt:', best_stat)
                self._materialize_historical_best(best_stat)

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            primary_metric = 'coco_eval_bbox'
            current_score = None
            previous_best = best_stat.get(primary_metric, float('-inf'))
            if primary_metric in test_stats:
                current_score = self._metric_value(test_stats[primary_metric])

            for k in test_stats.keys():
                score = self._metric_value(test_stats[k])
                best_stat[k] = max(best_stat.get(k, float('-inf')), score)

            is_best = current_score is not None and current_score > previous_best
            if is_best:
                best_stat['epoch'] = epoch

            self.best_stat = best_stat
            print('best_stat: ', best_stat)

            if is_best and self.output_dir:
                best_path = self.output_dir / 'best.pth'
                dist.save_on_master(self.state_dict(epoch), best_path)
                print(f'Saved new best checkpoint to {best_path} '
                      f'({primary_metric}: {current_score:.6f})')


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

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


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
