import os
from typing import Dict
from pathlib import Path
import torch
from torch import nn
import time
from tqdm.auto import trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler

from seggini.model import NR_CLASSES, BACKGROUND_CLASS, VARIABLE_SIZE, WSI_FIX, THRESHOLD, DISCARD_THRESHOLD
from seggini.model import GraphDataset
from seggini.model import prepare_graph_dataset, prepare_graph_dataloader, get_config, get_batched_segmentation_maps
from seggini.model import NodeClassifier
from seggini.model import get_loss_criterion, get_optimizer
from seggini.model import LoggingHelper
from .inference import test_classifier


def train_classifier(
        base_path: Path,
        data_config: Dict,
        model_config: Dict,
        metrics_config: Dict,
        params: Dict,
        **kwargs,
) -> nn.Module:
    """Train the classification model for a given number of epochs.
    """

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
    model = NodeClassifier(nr_classes=NR_CLASSES, **model_config)
    model = model.to(device)

    # Loss functions
    train_criterion = get_loss_criterion(params["loss"], train_dataset, supervision_mode="node", name="node", device=device)
    val_criterion = get_loss_criterion(params["loss"], val_dataset, supervision_mode="node", name="node", device=device)

    # Optimizer
    optim, scheduler = get_optimizer(params["optimizer"], model)

    # Metrics
    train_metric_logger = LoggingHelper(
        name="node",
        metrics_config=metrics_config,
        background_label=BACKGROUND_CLASS,
        nr_classes=NR_CLASSES,
        discard_threshold=DISCARD_THRESHOLD,
        threshold=THRESHOLD,
        wsi_fix=WSI_FIX,
        eval_segmentation=data_config["train_data"]["eval_segmentation"]
    )
    val_metric_logger = LoggingHelper(
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

    # Training loop
    for epoch in trange(params["num_epochs"]):
        start_time = time.time()

        # Train model
        model.train()
        for graph_batch in train_dataloader:
            optim.zero_grad()

            graph = graph_batch.meta_graph.to(device)
            targets = graph_batch.node_labels.to(device)
            logits = model(graph)

            # Calculate loss
            loss_information = {
                "logits": logits,
                "targets": targets,
                "node_associations": graph.batch_num_nodes,
            }
            loss = train_criterion(**loss_information)
            loss.backward()

            # Optimize
            if params['clip_gradient_norm'] is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), params['clip_gradient_norm']
                )
            optim.step()

            # Log metrics
            train_metric_logger.add_iteration_outputs(
                loss=loss.item(), **loss_information
            )

        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        train_metrics = train_metric_logger.log_and_clear()

        # Validate model
        val_metrics = None
        if epoch % params['validation_frequency'] == 0:
            model.eval()
            for graph_batch in val_dataloader:
                with torch.no_grad():
                    graph = graph_batch.meta_graph.to(device)
                    targets = graph_batch.node_labels.to(device)
                    logits = model(graph)

                    # Calculate loss
                    loss_information = {
                        "logits": logits,
                        "targets": targets,
                        "node_associations": graph.batch_num_nodes,
                    }
                    loss = val_criterion(**loss_information)

                if data_config["val_data"]["eval_segmentation"]:
                    segmentation_maps = get_batched_segmentation_maps(
                        node_logits=loss_information["logits"],
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

                val_metric_logger.add_iteration_outputs(
                    loss=loss.item(),
                    annotation=annotation,
                    predicted_segmentation=segmentation_maps,
                    tissue_masks=tissue_masks,
                    image_labels=graph_batch.graph_labels,
                    **loss_information,
                )

            val_metrics = val_metric_logger.log_and_clear(model=model)

            if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics[params["focused_metric"]])

        epoch_duration = time.time() - start_time
        print('epoch: ', epoch, ' -time: ', epoch_duration,
              ' -train_metrics: ', train_metrics,
              ' -val_metrics: ', val_metrics, '\n')

    return val_metric_logger.best_model


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