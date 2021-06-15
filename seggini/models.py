from typing import Dict, Tuple
import dgl
import torch
from torch import nn
from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN
from seggini.utils import *


class ClassifierHead(nn.Module):
    """A basic classifier head"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        hidden_dim: int = None,
        activation: str = "ReLU",
        input_dropout: float = 0.0,
        layer_dropout: float = 0.0,
    ) -> None:
        """Create a basic classifier head

        Args:
            input_dim (int): Dimensionality of the input
            output_dim (int): Number of output classes
            n_layers (int, optional): Number of layers (including input to hidden and hidden to output layer). Defaults to 2.
            hidden_dim (int, optional): Dimensionality of the hidden layers. Defaults to None.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        activation = dynamic_import_from("torch.nn", activation)
        modules = []
        if input_dropout > 0:
            modules.append(nn.Dropout(input_dropout))
        if n_layers > 1:
            modules.append(nn.Linear(input_dim, hidden_dim))
            modules.append(activation())
        for _ in range(n_layers - 2):
            if layer_dropout > 0:
                modules.append(nn.Dropout(layer_dropout))
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(activation())
        if n_layers == 1:
            modules.append(nn.Linear(input_dim, output_dim))
        else:
            modules.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass through the classifier head

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)


class NodeClassifierHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        node_classifier_config: Dict,
        nr_classes: int = NR_CLASSES,
    ) -> None:
        super().__init__()
        self.seperate_heads = node_classifier_config.pop("seperate_heads", True)
        if self.seperate_heads:
            node_classifiers = [
                ClassifierHead(
                    input_dim=latent_dim, output_dim=1, **node_classifier_config
                )
                for _ in range(nr_classes)
            ]
            self.node_classifiers = nn.ModuleList(node_classifiers)
        else:
            self.node_classifier = ClassifierHead(
                input_dim=latent_dim, output_dim=nr_classes, **node_classifier_config
            )

    def forward(self, node_embedding: torch.Tensor) -> torch.Tensor:
        if self.seperate_heads:
            node_logit = torch.empty(
                (node_embedding.shape[0], len(self.node_classifiers)),
                device=node_embedding.device,
            )
            for i, node_classifier in enumerate(self.node_classifiers):
                classifier_output = node_classifier(node_embedding).squeeze(1)
                node_logit[:, i] = classifier_output
            return node_logit
        else:
            return self.node_classifier(node_embedding)


class GraphClassifierHead(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            graph_classifier_config: Dict,
            nr_classes: int = NR_CLASSES
    ) -> None:
        super().__init__()
        self.graph_classifier = ClassifierHead(
            input_dim=latent_dim, output_dim=nr_classes, **graph_classifier_config,
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.graph_classifier(graph_embedding)


class GraphClassifier(nn.Module):
    """Classifier that uses graph labels and performs graph classification"""

    def __init__(
        self,
        nr_classes: int,
        gnn_config: Dict,
        graph_classifier_config: Dict,
    ) -> None:
        """
        Args:
            config (Dict): Configuration of the models
            nr_classes (int, optional): Number of classes to consider. Defaults to 4.
        """

        super().__init__()

        self.gnn_model = MultiLayerGNN(**gnn_config)
        if gnn_config["readout_op"] in ["none", "lstm"]:
            latent_dim = gnn_config["output_dim"]
        elif gnn_config["readout_op"] in ["concat"]:
            latent_dim = gnn_config["output_dim"] * gnn_config["num_layers"]
        else:
            raise NotImplementedError(
                f"Only supported readout op are [none, lstm, concat]"
            )
        self.graph_classifier = GraphClassifierHead(
            latent_dim, graph_classifier_config, nr_classes
        )

    def forward(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass on the graph
        Args:
            graph (dgl.DGLGraph): Input graph with node features in GNN_NODE_FEAT_IN
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits of the graph classifier, logits of the node classifiers
        """
        in_features = graph.ndata[GNN_NODE_FEAT_IN]
        graph_embedding = self.gnn_model(graph, in_features)
        graph_logit = self.graph_classifier(graph_embedding)
        return graph_logit


class NodeClassifier(nn.Module):
    """Classifier that uses node labels and performs node classification"""
    def __init__(
        self,
        gnn_config: Dict,
        node_classifier_config: Dict,
        nr_classes: int = NR_CLASSES,
    ) -> None:
        super().__init__()
        self.gnn_model = MultiLayerGNN(**gnn_config)
        if gnn_config["readout_op"] in ["none", "lstm"]:
            latent_dim = gnn_config["output_dim"]
        elif gnn_config["readout_op"] in ["concat"]:
            latent_dim = gnn_config["output_dim"] * gnn_config["num_layers"]
        else:
            raise NotImplementedError(
                f"Only supported agg operators are [none, lstm, concat]"
            )
        self.node_classifier = NodeClassifierHead(
            latent_dim, node_classifier_config, nr_classes
        )

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        in_features = graph.ndata[GNN_NODE_FEAT_IN]
        self.gnn_model(graph, in_features)
        node_embedding = graph.ndata[GNN_NODE_FEAT_OUT]
        node_logit = self.node_classifier(node_embedding)
        return node_logit


class CombinedClassifier(nn.Module):
    """Classifier that uses both graph labels and node labels and performs node classification"""

    def __init__(
        self,
        gnn_config: Dict,
        graph_classifier_config: Dict,
        node_classifier_config: Dict,
        nr_classes: int = NR_CLASSES,
    ) -> None:
        """Build a classifier to classify superpixel tissue graphs

        Args:
            config (Dict): Configuration of the models
            nr_classes (int, optional): Number of classes to consider. Defaults to 4.
        """
        super().__init__()
        self.gnn_model = MultiLayerGNN(**gnn_config)
        if gnn_config["readout_op"] in ["none", "lstm"]:
            latent_dim = gnn_config["output_dim"]
        elif gnn_config["readout_op"] in ["concat"]:
            latent_dim = gnn_config["output_dim"] * gnn_config["num_layers"]
        else:
            raise NotImplementedError(
                f"Only supported agg operators are [none, lstm, concat]"
            )
        self.graph_classifier = GraphClassifierHead(
            latent_dim, graph_classifier_config, nr_classes
        )
        self.node_classifiers = NodeClassifierHead(
            latent_dim, node_classifier_config, nr_classes
        )

    def forward(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass on the graph

        Args:
            graph (dgl.DGLGraph): Input graph with node features in GNN_NODE_FEAT_IN

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits of the graph classifier, logits of the node classifiers
        """
        in_features = graph.ndata[GNN_NODE_FEAT_IN]
        graph_embedding = self.gnn_model(graph, in_features)
        graph_logit = self.graph_classifier(graph_embedding)
        node_embedding = graph.ndata[GNN_NODE_FEAT_OUT]
        node_logit = self.node_classifiers(node_embedding)
        return graph_logit, node_logit


