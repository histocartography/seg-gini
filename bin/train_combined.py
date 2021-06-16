import os
from pathlib import Path
from typing import Optional, List, Dict
import time
from tqdm.auto import trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler
import copy
import torch
from torch import nn

from seggini.model import NR_CLASSES, BACKGROUND_CLASS, VARIABLE_SIZE, WSI_FIX, THRESHOLD, DISCARD_THRESHOLD
from seggini.model import GraphDataset
from seggini.model import prepare_graph_dataset, prepare_graph_dataloader, get_config, get_batched_segmentation_maps
from seggini.model import CombinedClassifier
from seggini.model import get_loss, get_optimizer
from seggini.model import LoggingHelper, BaseLogger
from .inference import test_classifier, GraphGradCAMBasedInference


class CombinedCriterion(torch.nn.Module):
    def __init__(self, loss: dict, device) -> None:
        super().__init__()
        self.graph_criterion = get_loss(loss, "graph", device)
        self.node_criterion = get_loss(loss, "node", device)
        self.node_loss_weight = loss.get("params", {}).get("node_weight", 0.5)
        assert (
            0.0 <= self.node_loss_weight <= 1.0
        ), f"Node weight loss must be between 0 and 1, but is {self.node_loss_weight}"
        self.graph_loss_weight = 1.0 - self.node_loss_weight
        self.device = device

    def forward(
        self,
        graph_logits: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
        node_logits: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
        node_associations: Optional[List[int]] = None,
    ):
        assert (
            graph_logits is not None and graph_labels is not None
        ), "Cannot use combined criterion without graph input"
        assert (
            node_logits is not None and node_labels is not None
        ), "Cannot use combined criterion without node input"
        node_labels = node_labels.to(self.device)
        graph_labels = graph_labels.to(self.device)

        graph_loss = self.graph_criterion(
            logits=graph_logits,
            targets=graph_labels
        )
        node_loss = self.node_criterion(
            logits=node_logits,
            targets=node_labels,
            node_associations=node_associations,
        )
        combined_loss = (
            self.graph_loss_weight * graph_loss + self.node_loss_weight * node_loss
        )
        return combined_loss, graph_loss.detach().cpu(), node_loss.detach().cpu()


def train_classifier(
        base_path: Path,
        data_config: Dict,
        model_config: Dict,
        metrics_config: Dict,
        params: Dict,
        **kwargs,
) -> nn.Module:

    # Data sets
    train_dataset: GraphDataset
    val_dataset: GraphDataset
    train_dataset = prepare_graph_dataset(base_path=base_path, mode="train", **data_config["train_data"], **params)
    val_dataset = prepare_graph_dataset(base_path=base_path, mode="val", **data_config["val_data"], **params)

    if params['balanced_sampling']:
        training_sample_weights = train_dataset.get_graph_size_weights()
        sampler = WeightedRandomSampler(
            training_sample_weights, len(train_dataset), replacement=True
        )
    else:
        sampler = None

    # Data loaders
    train_dataloader = prepare_graph_dataloader(dataset=train_dataset,
                                                shuffle=not params['balanced_sampling'],
                                                sampler=sampler,
                                                **params)
    params["batch_size"] = 1 \
        if (VARIABLE_SIZE and data_config["val_data"]["eval_segmentation"]) \
        else params["batch_size"]
    val_dataloader = prepare_graph_dataloader(dataset=val_dataset, **params)

    # Compute device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model = CombinedClassifier(nr_classes=NR_CLASSES, **model_config)
    model = model.to(device)

    # Loss functions
    train_loss = copy.deepcopy(params["loss"])
    if train_loss["params"]["use_weighted_loss"]:
        train_dataset.set_mode("node")
        train_loss["node"]["params"]["weight"] = train_dataset.get_dataset_loss_weights(
            log=train_loss["params"]["use_log_frequency_weights"]
        )
        train_dataset.set_mode("graph")
        train_loss["graph"]["params"]["weight"] = train_dataset.get_dataset_loss_weights(
            log=train_loss["params"]["use_log_frequency_weights"]
        )
        train_dataset.set_mode("node")
    train_criterion = CombinedCriterion(train_loss, device)

    val_loss = copy.deepcopy(params["loss"])
    if val_loss["params"]["use_weighted_loss"]:
        val_dataset.set_mode("node")
        val_loss["node"]["params"]["weight"] = val_dataset.get_dataset_loss_weights(
            log=val_loss["params"]["use_log_frequency_weights"]
        )
        val_dataset.set_mode("graph")
        val_loss["graph"]["params"]["weight"] = val_dataset.get_dataset_loss_weights(
            log=val_loss["params"]["use_log_frequency_weights"]
        )
        val_dataset.set_mode("node")
    val_criterion = CombinedCriterion(val_loss, device)

    # Optimizer
    optim, scheduler = get_optimizer(params["optimizer"], model)

    # Metrics
    train_graph_metric_logger = LoggingHelper(
        name="graph",
        metrics_config=metrics_config,
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
        eval_segmentation=False
    )
    train_node_metric_logger = LoggingHelper(
        name="node",
        metrics_config=metrics_config,
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
        eval_segmentation=False
    )
    train_combined_metric_logger = BaseLogger(
        metrics_config={},
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
    )
    val_graph_metric_logger = LoggingHelper(
        name="graph",
        metrics_config=metrics_config,
        focused_metric=params["focused_metric"],
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
        discard_threshold=DISCARD_THRESHOLD,
        threshold=THRESHOLD,
        wsi_fix=WSI_FIX,
        eval_segmentation=False
    )
    val_node_metric_logger = LoggingHelper(
        name="node",
        metrics_config=metrics_config,
        focused_metric=params["focused_metric"],
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
        discard_threshold=DISCARD_THRESHOLD,
        threshold=THRESHOLD,
        wsi_fix=WSI_FIX,
        eval_segmentation=data_config["val_data"]["eval_segmentation"]
    )
    val_combined_metric_logger = BaseLogger(
        metrics_config={},
        focused_metric=params["focused_metric"],
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
        discard_threshold=DISCARD_THRESHOLD,
        threshold=THRESHOLD,
        wsi_fix=WSI_FIX,
    )

    # Training loop
    for epoch in trange(params["num_epochs"]):
        start_time = time.time()

        # Train model
        model.train()
        for graph_batch in train_dataloader:
            optim.zero_grad()

            graph = graph_batch.meta_graph.to(device)
            graph_logits, node_logits = model(graph)

            # Calculate loss
            loss_information = {
                "graph_logits": graph_logits,
                "graph_labels": graph_batch.graph_labels.to(device),
                "node_logits": node_logits,
                "node_labels": graph_batch.node_labels.to(device),
                "node_associations": graph.batch_num_nodes,
            }
            combined_loss, graph_loss, node_loss = train_criterion(**loss_information)
            combined_loss.backward()

            # Optimize
            if params['clip_gradient_norm'] is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), params['clip_gradient_norm']
                )
            optim.step()

            # Log metrics
            train_graph_metric_logger.add_iteration_outputs(
                logits=loss_information["graph_logits"],
                targets=loss_information["graph_labels"],
            )
            train_node_metric_logger.add_iteration_outputs(
                logits=loss_information["node_logits"],
                targets=loss_information["node_labels"],
                node_associations=graph.batch_num_nodes,
            )
            train_combined_metric_logger.add_iteration_outputs(loss=combined_loss)

        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        train_metrics = train_graph_metric_logger.log_and_clear()
        train_metrics.update(train_node_metric_logger.log_and_clear())
        train_metrics.update(train_combined_metric_logger.log_and_clear())

        # Validate model
        val_metrics = None
        if epoch % params['validation_frequency'] == 0:
            model.eval()
            for graph_batch in val_dataloader:
                with torch.no_grad():
                    graph = graph_batch.meta_graph.to(device)
                    graph_logits, node_logits = model(graph)

                    # Calculate loss
                    loss_information = {
                        "graph_logits": graph_logits,
                        "graph_labels": graph_batch.graph_labels.to(device),
                        "node_logits": node_logits,
                        "node_labels": graph_batch.node_labels.to(device),
                        "node_associations": graph.batch_num_nodes,
                    }
                    combined_loss, graph_loss, node_loss = val_criterion(**loss_information)

                assert (
                        graph_batch.segmentation_masks is not None
                ), f"Cannot compute segmentation metrics if annotations are not loaded"

                # Graph Head Prediction
                if data_config["val_data"]["eval_segmentation"]:
                    inferencer = GraphGradCAMBasedInference(
                        NR_CLASSES, model, device=device
                    )
                    segmentation_maps = inferencer.predict_batch(
                        graph, graph_batch.instance_maps
                    )
                    annotation = torch.as_tensor(graph_batch.segmentation_masks)
                    tissue_masks = graph_batch.tissue_masks.astype(bool)
                else:
                    segmentation_maps = None
                    annotation = None
                    tissue_masks = None

                val_graph_metric_logger.add_iteration_outputs(
                    logits=loss_information["graph_logits"],
                    targets=loss_information["graph_labels"],
                    annotation=annotation,
                    predicted_segmentation=segmentation_maps,
                    tissue_masks=tissue_masks,
                    image_labels=graph_batch.graph_labels,
                )

                # Node Head Prediction
                if data_config["val_data"]["eval_segmentation"]:
                    segmentation_maps = get_batched_segmentation_maps(
                        node_logits=loss_information["node_logits"],
                        node_associations=graph.batch_num_nodes,
                        superpixels=graph_batch.instance_maps,
                        NR_CLASSES=NR_CLASSES,
                    )
                    segmentation_maps = torch.as_tensor(segmentation_maps)
                    annotation = torch.as_tensor(graph_batch.segmentation_masks)
                    tissue_masks = graph_batch.tissue_masks.astype(bool)
                else:
                    segmentation_maps = None
                    annotation = None
                    tissue_masks = None

                val_node_metric_logger.add_iteration_outputs(
                    logits=loss_information["node_logits"],
                    targets=loss_information["node_labels"],
                    annotation=annotation,
                    predicted_segmentation=segmentation_maps,
                    tissue_masks=tissue_masks,
                    image_labels=graph_batch.graph_labels,
                    node_associations=graph.batch_num_nodes,
                )

                val_combined_metric_logger.add_iteration_outputs(
                    loss=combined_loss
                )

            val_metrics = val_graph_metric_logger.log_and_clear(model=model)
            val_metrics.update(val_node_metric_logger.log_and_clear(model=model))
            val_metrics.update(val_combined_metric_logger.log_and_clear(model=model))

            if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics[params["focused_metric"]])

        epoch_duration = time.time() - start_time
        print('epoch: ', epoch, ' -time: ', epoch_duration,
              ' -train_metrics: ', train_metrics,
              ' -val_metrics: ', val_metrics, '\n')

    if val_graph_metric_logger.best_model is not None:
        return val_graph_metric_logger.best_model
    elif val_node_metric_logger.best_model is not None:
        return val_node_metric_logger.best_model
    elif val_combined_metric_logger.best_model is not None:
        return val_combined_metric_logger.best_model
    else:
        print('ERROR! best model is None')
        exit()


if __name__ == "__main__":
    base_path, config = get_config()

    # Train classifier
    model = train_classifier(
        base_path=base_path,
        data_config=config["train"]["data"],
        model_config=config["train"]["model"],
        metrics_config=config["train"]["metrics"],
        params=config["train"]["params"]
    )

    # Save model
    model_save_path = base_path / \
                      'models' / \
                      ('graph' +
                       '_partial_' + str(config["train"]["params"]["partial"]) +
                       '_fold_' + str(config["train"]["params"]["fold"]))
    os.makedirs(str(model_save_path), exist_ok=True)
    torch.save(model, model_save_path / "best_model.pt")

    # Test classifier
    prediction_save_path = base_path / \
                           'predictions' / \
                           ('graph' +
                            '_partial_' + str(config["train"]["params"]["partial"]) +
                            '_fold_' + str(config["train"]["params"]["fold"]))
    os.makedirs(str(prediction_save_path), exist_ok=True)

    # Test data set
    test_dataset = prepare_graph_dataset(base_path=base_path, mode="test", **config["test"]["data"]["test_data"])

    # segmentation, area based gleason grading
    test_classifier(
        model=model,
        test_dataset=test_dataset,
        prediction_save_path=prediction_save_path,
        **config["test"]["params"],
    )