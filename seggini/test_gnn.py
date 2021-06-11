import matplotlib.pyplot as plt
from abc import abstractmethod
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import dgl
from torch import nn
from typing import Callable
from functools import partial

from histocartography.interpretability import GraphGradCAMExplainer

from utils import *
from logger import BaseLogger
from metrics import (
    F1Score
)
from models import (
    GraphClassifier,
    NodeClassifier,
    CombinedClassifier
)
from dataloader import (
    GraphDataset,
    GraphDatapoint,
    collate_graphs
)


class BaseInference:
    def __init__(self, model, device=None, **kwargs) -> None:
        super().__init__()
        self.model = model.eval()
        if device is not None:
            self.device = device
        else:
            self.device = next(model.parameters()).device
        self.model = self.model.to(self.device)

    @abstractmethod
    def predict(*args, **kwargs):
        pass


# GNN Classification Inference
class ClassificationInference(BaseInference):
    def __init__(
            self, model, device, criterion=None
    ) -> None:
        super().__init__(model, device=device)
        if criterion is not None:
            self.criterion = criterion.to(self.device)
        else:
            self.criterion = None


class GraphBasedInference(ClassificationInference):
    def predict(self, dataset: GraphDataset, logger: BaseLogger):
        old_state = dataset.eval_segmentation
        dataset.eval_segmentation = False
        dataset_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=0,
        )
        with torch.no_grad():
            for graph_batch in tqdm(dataset_loader, total=len(dataset_loader)):
                graph = graph_batch.meta_graph.to(self.device)
                labels = graph_batch.graph_labels.to(self.device)
                logits = self.model(graph)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if self.criterion is not None:
                    loss_information = {
                        "logits": logits,
                        "targets": labels,
                    }
                    loss = self.criterion(**loss_information)
                else:
                    loss = None
                logger.add_iteration_outputs(
                    loss=loss, 
                    logits=logits, 
                    targets=labels)
        metrics = logger.log_and_clear()
        dataset.eval_segmentation = old_state
        return metrics


class NodeBasedInference(ClassificationInference):
    def predict(self, dataset: GraphDataset, logger: BaseLogger):
        old_state = dataset.eval_segmentation
        dataset.eval_segmentation = False
        dataset_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=0,
        )
        with torch.no_grad():
            for graph_batch in tqdm(dataset_loader, total=len(dataset_loader)):
                graph = graph_batch.meta_graph.to(self.device)
                labels = graph_batch.node_labels.to(self.device)
                logits = self.model(graph)
                if isinstance(logits, tuple):
                    logits = logits[1]
                if self.criterion is not None:
                    loss_information = {
                        "logits": logits,
                        "targets": labels,
                        "node_associations": graph.batch_num_nodes,
                    }
                    loss = self.criterion(**loss_information)
                else:
                    loss = None
                logger.add_iteration_outputs(
                    loss=loss,
                    logits=logits,
                    targets=labels,
                    node_associations=graph.batch_num_nodes,
                )
        metrics = logger.log_and_clear()
        dataset.eval_segmentation = old_state
        return metrics
    

# GNN Segmentation Inference
class GraphGradCAMBasedInference(BaseInference):
    def __init__(self, NR_CLASSES, model, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.NR_CLASSES = NR_CLASSES
        self.explainer = GraphGradCAMExplainer(model=model)

    def predict(self, graph, superpixels, operation="argmax"):
        assert operation == "argmax"

        graph = graph.to(self.device)
        importances, logits = self.explainer.process(
            graph, list(range(self.NR_CLASSES))
        )
        node_importances = (
                importances * torch.as_tensor(logits)[0].sigmoid().numpy()[:, np.newaxis]
        ).argmax(0)
        return get_segmentation_map(
            node_predictions=node_importances,
            superpixels=superpixels,
            NR_CLASSES=self.NR_CLASSES)

    def predict_batch(self, graphs, superpixels, operation="argmax"):
        segmentation_maps = list()
        for i, graph in enumerate(dgl.unbatch(graphs)):
            segmentation_map = self.predict(graph, superpixels[i], operation)
            segmentation_maps.append(segmentation_map)
        return np.stack(segmentation_maps)


class GraphNodeBasedInference(BaseInference):
    def __init__(self, model, device, NR_CLASSES, **kwargs) -> None:
        super().__init__(model, device=device, **kwargs)
        self.NR_CLASSES = NR_CLASSES

    def predict(self, graph, superpixels, operation="argmax"):
        assert operation == "argmax"

        graph = graph.to(self.device)
        node_logits = self.model(graph)
        if isinstance(node_logits, tuple):
            node_logits = node_logits[1]

        node_predictions = node_logits.argmax(axis=1).detach().cpu().numpy()
        return get_segmentation_map(
            node_predictions=node_predictions,
            superpixels=superpixels,
            NR_CLASSES=self.NR_CLASSES,
        )


# Dataset Segmentation Inferencer

class DatasetBaseInference:
    def __init__(
            self, inferer: BaseInference, callbacks: Optional[Callable] = []
    ) -> None:
        self.inferer = inferer
        self.callbacks = callbacks

    @abstractmethod
    def _handle_datapoint(
            self,
            datapoint: Any,
            operation: str,
            logger: Optional[BaseLogger] = None
    ):
        pass

    def __call__(
            self,
            dataset: Dataset,
            logger: Optional[BaseLogger] = None,
            **kwargs,
    ):
        for i in tqdm(range(len(dataset))):
            datapoint = dataset[i]
            prediction = self._handle_datapoint(
                datapoint=datapoint,
                logger=logger,
                **kwargs,
            )
            for callback in self.callbacks:
                callback(prediction=prediction, datapoint=datapoint)

        if logger is not None:
            metrics = logger.log_and_clear()
            print("Metrics: ", metrics)


class GraphDatasetInference(DatasetBaseInference):
    def _handle_datapoint(
            self,
            datapoint: GraphDatapoint,
            operation: str,
            logger: Optional[BaseLogger] = None
    ):
        prediction = self.inferer.predict(
            datapoint.graph,
            datapoint.instance_map,
            operation=operation,
        )
        if logger is not None:
            logger.add_iteration_outputs(
                logits=prediction.copy()[np.newaxis, ...],
                targets=datapoint.segmentation_mask[np.newaxis, ...],
                tissue_mask=datapoint.tissue_mask.astype(bool)[np.newaxis, ...],
                image_labels=datapoint.graph_label[np.newaxis, ...],
            )
        return prediction


# Helper function to save segmentation masks
def log_segmentation_mask(
        prediction: np.ndarray,
        datapoint: GraphDatapoint,
        operation: str,
        save_path: Path
):
    tissue_mask = datapoint.tissue_mask
    prediction[~tissue_mask.astype(bool)] = BACKGROUND_CLASS
    ground_truth = datapoint.segmentation_mask.copy()
    ground_truth[~tissue_mask.astype(bool)] = BACKGROUND_CLASS

    # Save figure
    if operation == "per_class":
        fig = show_class_activation(prediction)
    elif operation == "argmax":
        fig = show_segmentation_masks(
            prediction,
            annotation=ground_truth
        )
    else:
        raise NotImplementedError(
            f"Only support operation [per_class, argmax], but got {operation}"
        )

    # Set title to be DICE score
    metric = F1Score(
        nr_classes=NR_CLASSES,
        discard_threshold=DISCARD_THRESHOLD,
        background_label=BACKGROUND_CLASS,
    )
    metric_value = metric(prediction=[prediction], ground_truth=[ground_truth])
    fig.suptitle(
        f"Benign: {metric_value[0]}, Grade 3: {metric_value[1]}, Grade 4: {metric_value[2]}, Grade 5: {metric_value[3]}"
    )

    file_name = save_path / f"{datapoint.name}.png"
    fig.savefig(str(file_name), dpi=300, bbox_inches="tight")
    plt.close(fig=fig)

def test_classifier(
        model: nn.Module,
        test_dataset: Dataset,
        prediction_save_path: Path,
        operation: str = "argmax",
        use_grad_cam: bool = False,
        **kwargs,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get mode
    if isinstance(model, GraphClassifier):
        mode = "graph_supervision"
    elif isinstance(model, NodeClassifier):
        mode = "node_supervision"
    elif isinstance(model, CombinedClassifier):
        mode = "combined_supervision"
    else:
        raise NotImplementedError

    # Classification Inference
    if mode in ["graph_supervision", "combined_supervision"]:
        classification_inferer = GraphBasedInference(model=model, device=device)
        graph_logger = BaseLogger(
            [
                "MultiLabelF1Score",
            ],
            nr_classes=NR_CLASSES,
            background_label=BACKGROUND_CLASS,
        )
        classification_metrics = classification_inferer.predict(test_dataset, graph_logger)
    if mode in ["node_supervision", "combined_supervision"]:
        classification_inferer = NodeBasedInference(model=model, device=device)
        node_logger = BaseLogger(
            [
                "NodeClassificationF1Score"
            ],
            nr_classes=NR_CLASSES,
            background_label=BACKGROUND_CLASS,
        )
        classification_metrics = classification_inferer.predict(test_dataset, node_logger)
    print("Classification metrics: ", classification_metrics)


    # Segmentation Inference
    if mode == "graph_supervision" or \
            (mode == "combined_supervision" and use_grad_cam):
        inferer = GraphGradCAMBasedInference(
            model=model,
            device=device,
            NR_CLASSES=NR_CLASSES,
            **kwargs
        )
    else:
        inferer = GraphNodeBasedInference(
            model=model,
            device=device,
            NR_CLASSES=NR_CLASSES,
            **kwargs,
        )
    
    logger_pathologist = BaseLogger(
        [
            "GleasonScoreF1",
            "DatasetDice",
        ],
        callbacks=[
            partial(
                save_confusion_matrix,
                classes=["Benign", "Grade6", "Grade7", "Grade8", "Grade9", "Grade10"],
                save_path=prediction_save_path / "GleasonScoreF1.png",
            )
        ],
        nr_classes=NR_CLASSES,
        background_label=BACKGROUND_CLASS,
        variable_size=VARIABLE_SIZE,
        wsi_fix=WSI_FIX,
        threshold=THRESHOLD,
        discard_threshold=DISCARD_THRESHOLD,
        enabled_callbacks=True
    )

    inference_runner = GraphDatasetInference(
        inferer=inferer,
        callbacks=[
            partial(
                log_segmentation_mask,
                operation=operation,
                save_path=prediction_save_path,
            )
        ]
    )

    inference_runner(
        dataset=test_dataset,
        logger=logger_pathologist,
        operation=operation,
    )