from dataclasses import asdict, dataclass
from typing import Tuple, List, Callable
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import h5py
import dgl
import torch
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from metrics import inverse_frequency, inverse_log_frequency
from utils import *

class BaseDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        image_label_mapper: dict,
        supervision_mode: str = "graph",
        eval_segmentation: bool = False,
        downsample: Optional[int] = 1,
    ) -> None:
        """
        Args:
            metadata (pd.DataFrame): Image metadata
            image_label_mapper (dict): Image level labels
            supervision_mode (str): Provided supervision type
            num_classes (int): Number of tumor classes
            downsample (int): Downsample rate for image, tissue mask, annotation mask

        Return:
            names (np.ndarray): Names of images in metadata
            graphs (np.ndarray): Graphs of images in metadata
            image_sizes (np.ndarray): Dimension of images in metadata
            annotations (np.ndarray): Annotation masks of images in metadata
            tissue_masks (np.ndarray): Tissue masks of images in metadata
            graph_labels (np.ndarray): Image label of images in metadata
            image_indices (np.ndarray): Indices of images in metadata
        """
        assert supervision_mode in [
            "graph",
            "node",
        ], f"supervision_mode must be in [graph, node] but is {supervision_mode}"

        self.metadata = metadata
        self.image_label_mapper = image_label_mapper

        self.supervision_mode = supervision_mode
        self.num_classes = NR_CLASSES
        self.downsample = downsample

        self.eval_segmentation = eval_segmentation
        self._load()
        self._image_indices = np.arange(0, len(self._graphs))

    def _load(self):
        self._initalize_loading()
        for i, row in tqdm(
                self.metadata.iterrows(), total=len(self.metadata), desc=f"Dataset Loading"
        ):
            self._load_datapoint(i, row)
        self._finish_loading()

    def _initalize_loading(self):
        self._names = list()
        self._graphs = list()
        self._image_sizes = list()
        self._annotations = list()
        self._superpixels = list()
        self._tissue_masks = list()
        self._graph_labels = list()

    def _load_datapoint(self, i, row):
        self._names.append(self._load_name(i, row))
        self._graphs.append(self._load_graph(i, row))
        self._graph_labels.append(self._load_graph_label(i, row))
        self._image_sizes.append(self._load_image_size(i, row))

        if self.eval_segmentation:
            self._annotations.append(self._load_image(row["annotation_mask_path"]))
            self._superpixels.append(self._load_h5(row["superpixel_path"]))
            self._tissue_masks.append(self._load_image(row["tissue_mask_path"]))

    def _finish_loading(self):
        self._names = np.array(self._names)
        self._graphs = np.array(self._graphs)
        self._image_sizes = np.array(self._image_sizes)
        self._annotations = np.array(self._annotations)
        self._superpixels = np.array(self._superpixels)
        self._tissue_masks = np.array(self._tissue_masks)
        self._graph_labels = np.array(self._graph_labels)

    @staticmethod
    def _load_name(i, row):
        return i

    @staticmethod
    def _load_graph(i, row):
        graph = load_graphs(str(row["graph_path"]))[0][0]
        graph.readonly()
        return graph

    def _load_h5(self, path):
        try:
            with h5py.File(path, "r") as file:
                if "default_key_0" in file:
                    content = file["default_key_0"][()]
                elif "default_key" in file:
                    content = file["default_key"][()]
                else:
                    raise NotImplementedError(
                        f"Superpixels not found in keys. Available are: {file.keys()}"
                    )
        except OSError as e:
            print(f"Could not open {path}")
            raise e
        return self._downsample(content)

    def _load_image(self, path):
        image = read_image(path)
        return self._downsample(image)

    @staticmethod
    def _load_image_size(i, row):
        return (row.height, row.width)

    def _load_graph_label(self, i, row):
        return self.image_label_mapper[i]

    def _downsample(self, array):
        if self.downsample != 1:
            new_size = (
                int(array.shape[0] // self.downsample),
                int(array.shape[1] // self.downsample),
            )
            array = cv2.resize(
                array,
                new_size,
                interpolation=cv2.INTER_NEAREST,
            )
        return array

    @property
    def names(self):
        return self._names

    @property
    def graphs(self):
        return self._graphs

    @property
    def image_sizes(self):
        return self._image_sizes

    @property
    def annotations(self):
        return self._annotations

    @property
    def superpixels(self):
        return self._superpixels

    @property
    def tissue_masks(self):
        return self._tissue_masks

    @property
    def graph_labels(self):
        return self._graph_labels


class GraphDataset(BaseDataset):
    """Dataset used for extracted and dumped graphs"""

    def __init__(
        self,
        node_dropout: Optional[float] = 0.0,
        **kwargs,
    ) -> None:
        """
        Args:
            node_dropout (float): Percentage of nodes to drop (augmentation)
        Return:
            name_to_index (dict): Maps image name to index
            node_weights (np.ndarray): Class-wise node weights for each image
        """

        super().__init__(**kwargs)
        self.node_dropout = node_dropout

        self._name_to_index = dict(zip(self._names, range(len(self._names))))
        self._graph_labels = torch.as_tensor(self._graph_labels)
        self._node_weights = self._compute_node_weights()

    def _compute_node_weight(self, node_labels: torch.Tensor) -> np.ndarray:
        class_counts = fast_histogram(node_labels, nr_values=self.num_classes)
        return inverse_log_frequency(class_counts.astype(np.float32)[np.newaxis, :])[0]

    def _compute_node_weights(self) -> np.ndarray:
        node_weights = list()
        for graph in self._graphs:
            if self.supervision_mode == "node":
                node_labels = graph.ndata[LABEL]
                node_weights.append(self._compute_node_weight(node_labels))
            else:
                node_weights.append(None)
        return np.array(node_weights)

    @property
    def names(self):
        return self._names

    @property
    def graphs(self):
        return self._graphs

    @property
    def image_sizes(self):
        return self._image_sizes

    @property
    def annotations(self):
        return self._annotations

    @property
    def superpixels(self):
        return self._superpixels

    @property
    def tissue_masks(self):
        return self._tissue_masks

    @property
    def graph_labels(self):
        return self._graph_labels

    @property
    def node_weights(self):
        return self._node_weights

    @property
    def indices(self):
        return self._image_indices

    @staticmethod
    def _generate_subgraph(
        graph: dgl.DGLGraph, node_indices: torch.Tensor
    ) -> dgl.DGLGraph:
        """Generates a subgraph with only the nodes and edges in node_indices

        Args:
            graph (dgl.DGLGraph): Input graph
            node_indices (torch.Tensor): Node indices to consider

        Returns:
            dgl.DGLGraph: A subgraph with the subset of nodes and edges
        """
        subgraph = graph.subgraph(node_indices)
        for key, item in graph.ndata.items():
            subgraph.ndata[key] = item[subgraph.ndata["_ID"]]
        for key, item in graph.edata.items():
            subgraph.edata[key] = item[subgraph.edata["_ID"]]
        return subgraph

    def _get_node_dropout_subgraph(
        self, graph: dgl.DGLGraph, node_labels: torch.Tensor
    ) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        node_mask = torch.rand(graph.number_of_nodes()) > self.node_dropout
        subgraph = self._generate_subgraph(graph, torch.where(node_mask)[0])
        if node_labels is not None:
            subgraph_node_labels = node_labels[node_mask]
        else:
            subgraph_node_labels = None
        assert (
            node_labels is None or subgraph.number_of_nodes() == subgraph_node_labels.shape[0]
        ), f"Dropout graph node labels do not correspond to the number of nodes. Graph size: {subgraph.number_of_nodes()}, labels: {subgraph_node_labels.shape}"
        return subgraph, subgraph_node_labels

    def set_mode(self, mode):
        valid_modes = ["graph", "node"]
        assert (
            mode in valid_modes
        ), f"Dataset mode must be from {valid_modes}, but is {mode}"
        self.supervision_mode = mode

    def get_labels(self) -> torch.Tensor:
        if self.supervision_mode == "node":
            node_labels = list()
            nr_datapoints = self.__len__()
            for i in range(nr_datapoints):
                datapoint = self.__getitem__(i)
                node_labels.append(datapoint.node_labels)
            return torch.cat(node_labels)
        elif self.supervision_mode == "graph":
            graph_labels = list()
            nr_datapoints = self.__len__()
            for i in range(nr_datapoints):
                datapoint = self.__getitem__(i)
                graph_labels.append(datapoint.graph_label)
            return torch.stack(graph_labels)
        else:
            raise NotImplementedError

    def get_dataset_loss_weights(self, log=True) -> torch.Tensor:
        labels = self.get_labels()
        if self.supervision_mode == "node":
            class_counts = fast_histogram(labels, self.num_classes)
        else:
            class_counts = labels.sum(dim=0).numpy()
        if log:
            class_weights = inverse_log_frequency(
                class_counts.astype(np.float32)[np.newaxis, :]
            )[0]
        else:
            class_weights = inverse_frequency(
                class_counts.astype(np.float32)[np.newaxis, :]
            )[0]
        return torch.as_tensor(class_weights)

    def get_graph_size_weights(self) -> torch.Tensor:
        nr_nodes = list()
        for graph in self.graphs:
            nr_nodes.append(graph.number_of_nodes())
        nr_nodes = np.array(nr_nodes)
        nr_nodes = nr_nodes / nr_nodes.sum()
        return torch.as_tensor(nr_nodes)

    def _build_datapoint(self, graph, node_labels, index):
        if self.supervision_mode == "node":
            return GraphDatapoint(
                name=self.names[index],
                graph=graph,
                graph_label=self.graph_labels[index],
                node_labels=node_labels,
                instance_map=self.superpixels[index]
                if self.eval_segmentation
                else None,
                segmentation_mask=self.annotations[index]
                if self.eval_segmentation
                else None,
                tissue_mask=self.tissue_masks[index]
                if self.eval_segmentation
                else None,
            )
        elif self.supervision_mode == "graph":
            assert node_labels is None
            return GraphDatapoint(
                name=self.names[index],
                graph=graph,
                graph_label=self.graph_labels[index],
                node_labels=None,
                instance_map=None,
                segmentation_mask=None,
                tissue_mask=None,
            )

    def __len__(self) -> int:
        """Number of graphs in the dataset
        Returns:
            int: Length of the dataset
        """
        assert len(self.graphs) == len(self._image_indices)
        return len(self.graphs)


class AugmentedGraphDataset(GraphDataset):
    """Dataset variation used for extracted and dumped graphs with augmentations"""

    def __init__(
            self,
            augmentation_mode: Optional[str] = None,
            centroid_features_mode: str = "no",
            **kwargs) -> None:
        assert augmentation_mode in [
            None,
            "node",
            "graph",
        ], f"centroid_features must be in [no, cat, only] but is {augmentation_mode}"
        assert centroid_features_mode in [
            "no",
            "cat",
            "only",
        ], f"centroid_features must be in [no, cat, only] but is {centroid_features_mode}"
        self.augmentation_mode = augmentation_mode
        self.centroid_features_mode = centroid_features_mode
        super().__init__(**kwargs)

    def _get_graph_features(self, graph, image_size) -> None:
        """Skips the feature selection step as it is done during data loading

        Args:
            graph (dgl.DGLGraph): Source graph
            image_size (np.ndarray): Image dimensions
        """
        if self.centroid_features_mode == "only":
            features = (graph.ndata[CENTROID] / torch.Tensor(image_size)).to(
                torch.float32
            )
        else:
            assert (
                len(graph.ndata[FEATURES].shape) == 3
            ), f"Cannot use AugmentedDataset when the preprocessing was not run with augmentations"
            nr_nodes, nr_augmentations, _ = graph.ndata[FEATURES].shape

            # Sample based on augmentation mode
            if self.augmentation_mode == "graph":
                sample = torch.ones(size=(nr_nodes,), dtype=torch.long) * torch.randint(
                    low=0, high=nr_augmentations, size=(1,)
                )
            elif self.augmentation_mode == "node":
                sample = torch.randint(
                    low=0, high=nr_augmentations, size=(nr_nodes,), dtype=torch.long
                )
            elif (
                isinstance(self.augmentation_mode, str)
                and self.augmentation_mode.isnumeric()
            ):
                sample = torch.ones(size=(nr_nodes,), dtype=torch.long) * int(
                    self.augmentation_mode
                )
            else:
                sample = torch.zeros(size=(nr_nodes,), dtype=torch.long)

            # Select features to use
            features = graph.ndata[FEATURES][torch.arange(nr_nodes), sample].to(
                torch.float32
            )

            if self.centroid_features_mode == "cat":
                features = torch.cat(
                    [
                        features,
                        (graph.ndata[CENTROID] / torch.Tensor(image_size)).to(
                            torch.float32
                        ),
                    ],
                    dim=1,
                )
        return features

    def __getitem__(self, index: int) -> Any:
        """Get a graph to train with. Randomly samples features from the available features.

        Args:
            index (int): Index of the graph

        Returns:
            Any: Return tuple depending on the arguments
        """
        if isinstance(index, str):
            index = self._name_to_index[index]
        assert (
            index in self.indices
        ), f"Index ({index}) not in range of datapoints ({self.indices}))."
        image_size = self.image_sizes[index]
        graph = self.graphs[index]

        # Create graph
        augmented_graph = dgl.DGLGraph(graph_data=graph)
        augmented_graph.ndata[CENTROID] = graph.ndata[CENTROID]

        # Set features
        augmented_graph.ndata[GNN_NODE_FEAT_IN] = self._get_graph_features(graph, image_size)

        # Set node labels
        if self.supervision_mode == "node":
            augmented_graph.ndata[LABEL] = graph.ndata[LABEL]
            node_labels = augmented_graph.ndata[LABEL]
        else:
            node_labels = None

        # Node dropout augmentation
        if self.node_dropout > 0:
            augmented_graph, node_labels = self._get_node_dropout_subgraph(graph, node_labels)

        return self._build_datapoint(augmented_graph, node_labels, index)


@dataclass
class GraphDatapoint:
    """Dataclass that holds a datapoint for a graph"""

    name: Optional[str] = None
    graph: dgl.DGLGraph = None
    graph_label: Optional[torch.IntTensor] = None
    node_labels: Optional[torch.IntTensor] = None
    instance_map: Optional[np.ndarray] = None
    segmentation_mask: Optional[np.ndarray] = None
    tissue_mask: Optional[np.ndarray] = None


@dataclass
class GraphBatch:
    """Dataclass for a batch of GraphDatapoints"""

    names: Optional[List[str]] = None
    meta_graph: dgl.DGLGraph = None
    graph_labels: Optional[torch.IntTensor] = None
    node_labels: Optional[torch.IntTensor] = None
    instance_maps: Optional[np.ndarray] = None
    segmentation_masks: Optional[np.ndarray] = None
    tissue_masks: Optional[np.ndarray] = None


def collate_graphs(samples: List[GraphDatapoint]) -> GraphBatch:
    """Turns a list of GraphDatapoint into a GraphBatch

    Args:
        samples (List[GraphDatapoint]): Input datapoints

    Returns:
        GraphBatch: Output batch
    """
    merged_datapoints = defaultdict(list)
    for sample in samples:
        for attribute, value in asdict(sample).items():
            if value is not None:
                merged_datapoints[attribute].append(value)

    nr_datapoints = len(samples)
    for attribute, values in merged_datapoints.items():
        assert (
            len(values) == nr_datapoints
        ), f"Could not batch samples, inconsistent attibutes: {samples}"

    def map_name(name: str):
        """Maps names of GraphDatapoint to the names of GraphBatch

        Args:
            name ([str]): Name of GraphDatapoint

        Returns:
            [str]: Name of GraphBatch
        """
        if name == "graph":
            return "meta_graph"
        elif name == "node_labels":
            return "node_labels"
        else:
            return name + "s"

    def merge(name: str, values: Any) -> Any:
        """Merges attributes based on the names

        Args:
            name (str): Name of attibute
            values (Any): Values to merge

        Returns:
            Any: Merged values
        """
        return {
            "graph": dgl.batch,
            "graph_label": torch.stack,
            "node_labels": torch.cat,
        }.get(name, np.stack)(values)

    return GraphBatch(
        **{map_name(k): merge(k, v) for k, v in merged_datapoints.items()}
    )

def prepare_graph_dataset(
    mode: str,
    fold: Optional[int] = -1,
    partial: Optional[int] = 100,
    supervision_mode: str = None,
    downsample: Optional[int] = 1,
    centroid_features_mode: str = "no",
    eval_segmentation: bool = False,
    augmentation_mode: Optional[str] = None,
    node_dropout: Optional[float] = 0.0,
    **kwargs
) -> Dataset:

    # Get paths and global variables
    constants = Constants(mode=mode, fold=fold, partial=partial)

    # Load all sample info
    all_metadata, label_mapper = get_metadata(constants)

    # Select subset of samples
    sample_names = []
    for id_path in constants.ID_PATHS:
        sample_names.append(pd.read_csv(id_path, index_col=0)["image_id"].values)
    sample_names = np.concatenate(sample_names)
    metadata = all_metadata.loc[
        list(set(sample_names) & set(all_metadata.index.values))
    ]

    # Set dataset arguments
    arguments = {
        "metadata": metadata,
        "image_label_mapper": label_mapper,
        "supervision_mode": supervision_mode,
        "downsample": downsample,
        "centroid_features_mode": centroid_features_mode,
        "eval_segmentation": eval_segmentation,
        "augmentation_mode": augmentation_mode,
        "node_dropout": node_dropout
    }
    return AugmentedGraphDataset(**arguments)

def prepare_graph_dataloader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        sampler: Optional = None,
        collate_fn: Optional[Callable] = collate_graphs,
        num_workers: Optional[int] = 0,
        **kwargs
) -> DataLoader:
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return dataloader