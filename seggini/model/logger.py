from typing import DefaultDict
import torch
import numpy as np

from .metrics import Metric
from .utils import dynamic_import_from

class BaseLogger:
    def __init__(
        self, metrics_config, focused_metric=None, variable_size=False, **kwargs
    ) -> None:
        self.metrics_config = metrics_config
        self.focused_metric = focused_metric
        self.variable_size = variable_size
        self._reset_epoch_stats()
        
        self.best_metric_values = list()
        self.best_model = None

        self.metric_names = list()
        self.metrics = list()
        for metric in metrics_config:
            metric_class = dynamic_import_from("seggini.model.metrics", metric)
            self.metrics.append(metric_class(**kwargs))
            self.metric_names.append(metric)
            self.best_metric_values.append(
                -float("inf") if metric_class.is_better(1, 0) else float("inf")
            )

    def _reset_epoch_stats(self):
        self.losses = list()
        self.logits = list()
        self.targets = list()
        self.extra_info = DefaultDict(list)

    def _compute_metric(self):
        if len(self.logits) == 0 or len(self.targets) == 0:
            return list()
        if not self.variable_size:
            logits = torch.cat(self.logits)
            targets = torch.cat(self.targets)
        else:
            logits = [item for sublist in self.logits for item in sublist]
            targets = [item for sublist in self.targets for item in sublist]
        metric_values = list()
        name: str
        metric: Metric
        for name, metric in zip(self.metric_names, self.metrics):
            try:
                metric_value = metric(logits, targets, **self.extra_info)
            except TypeError as e:
                print(
                    f"Got type error in metric {metric.__class__.__name__} __call__ function"
                )
                print(f"extra_info passed: {self.extra_info}")
                raise e
            if metric.is_per_class:
                metric_value[metric_value != metric_value] = 0.0
                mean_metric_value = np.nanmean(metric_value)
                metric_values.append(mean_metric_value)
            else:
                metric_values.append(metric_value)
        return metric_values

    def add_iteration_outputs(self, loss=None, logits=None, targets=None, **kwargs):
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self.losses.append(loss)
        if logits is not None:
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().cpu()
            else:
                logits = torch.as_tensor(logits)
            self.logits.append(logits)
        if targets is not None:
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu()
            else:
                targets = torch.as_tensor(targets)
            self.targets.append(targets)
        for name, value in kwargs.items():
            self.extra_info[name].extend(value)

    def log_and_clear(self, model=None):
        return_metric = {}
        
        if len(self.losses) > 0:
            return_metric["loss"] = np.mean(self.losses)

        if len(self.logits) > 0:
            current_metric_values = self._compute_metric()
            all_information = zip(
                self.metric_names,
                self.metrics,
                self.best_metric_values,
                current_metric_values,
            )
            metric: Metric
            for i, (name, metric, best_metric_value, current_metric_value) in enumerate(all_information):
                if metric.is_better(current_metric_value, best_metric_value):
                    self.best_metric_values[i] = current_metric_value

                    if name == self.focused_metric and metric.logs_model and model is not None:
                        self.best_model = model

            current_metrics = dict(zip(self.metric_names, current_metric_values))
            return_metric.update(current_metrics)

        self._reset_epoch_stats()
        return return_metric


class SegmentationLogger(BaseLogger):
    def __init__(
        self, metrics_config, background_label, **kwargs
    ) -> None:
        kwargs["background_label"] = background_label
        super().__init__(metrics_config, **kwargs)
        
        self.background_label = background_label
        self.variable_size = True

    def _reset_epoch_stats(self):
        self.masks = list()
        super()._reset_epoch_stats()

    def add_iteration_outputs(self, logits, targets, mask=None, loss=None, **kwargs):
        assert logits.shape == targets.shape, f"{logits.shape}, {targets.shape}"
        assert targets.shape == mask.shape, f"{targets.shape}, {mask.shape}"
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu()
            self.masks.append(mask)
        return super().add_iteration_outputs(
            loss=loss,
            logits=logits,
            targets=targets,
            tissue_mask=list(mask) if mask is not None else None,
            **kwargs,
        )

    def log_and_clear(self, model=None):
        return super().log_and_clear(model=model)


class LoggingHelper:
    def __init__(self, name, metrics_config, eval_segmentation, **kwargs) -> None:
        self.eval_segmentation = eval_segmentation
        self.classification_helper = BaseLogger(metrics_config.get(name, {}), **kwargs)

        if self.eval_segmentation:
            self.segmentation_helper = SegmentationLogger(metrics_config.get("segmentation", {}), **kwargs)

    def add_iteration_outputs(
        self,
        loss=None,
        logits=None,
        targets=None,
        annotation=None,
        predicted_segmentation=None,
        tissue_masks=None,
        **kwargs,
    ):
        self.classification_helper.add_iteration_outputs(
            loss=loss, logits=logits, targets=targets, **kwargs
        )
        if (
            predicted_segmentation is not None
            and annotation is not None
            and tissue_masks is not None
            and self.eval_segmentation
        ):
            self.segmentation_helper.add_iteration_outputs(
                logits=predicted_segmentation,
                targets=annotation,
                mask=tissue_masks,
                **kwargs,
            )

    def log_and_clear(self, model=None):
        classification_dict = self.classification_helper.log_and_clear(model)

        segmentation_dict = None
        if self.eval_segmentation:
            segmentation_dict = self.segmentation_helper.log_and_clear(model)

        self.best_model = self.classification_helper.best_model
        if classification_dict is None and segmentation_dict is not None:
            return_dict = segmentation_dict
            self.best_model = self.segmentation_helper.best_model
        if classification_dict is not None and segmentation_dict is None:
            return_dict = classification_dict
        if classification_dict is not None and segmentation_dict is not None:
            classification_dict.update(segmentation_dict)
            return_dict = classification_dict
            if self.segmentation_helper.best_model is not None:
                self.best_model = self.segmentation_helper.best_model

        return return_dict