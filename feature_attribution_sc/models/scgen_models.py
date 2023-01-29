import anndata as ad
import scgen
import torch
from scgen._base_components import DecoderSCGEN
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.nn import Encoder

from feature_attribution_sc.explainers.mask import mask


def int_to_str_map(y, labels):
    """The inference step of SCANVI provides the labels (perturbations, cell type) as an integer.

    The masking method requires these labels as strings of the perturbation or cell type, as the importance mapping may shuffle them.
    This method maps the integer labels to string labels.

    :param y: int labels of batch
    :param labels: all str labels
    :return: str labels of batch
    """
    try:
        str_labels = [labels[i] for i in y.tolist()]
    except IndexError:
        str_labels = labels
    return str_labels


class SCGENVAECustom(scgen.SCGENVAE):
    """This class inherits the original SCGENVAE class and overwrites the initialization, inference, and get inference input functions.

    Feature importance and thresholds are passed to the inference step, which calls the application
    of the mask, and now passes a masked datapoint.
    """

    def __init__(
        self,
        n_input: int,
        feature_importance: str,
        threshold: float,
        n_hidden: int = 800,
        n_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        kl_weight: float = 0.00005,
        labels=None,
    ):
        super(scgen.SCGENVAE, self).__init__()
        self.n_layers = n_layers
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = "normal"
        self.kl_weight = kl_weight

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            activation_fn=torch.nn.LeakyReLU,
        )

        n_input_decoder = n_latent
        self.decoder = DecoderSCGEN(
            n_input_decoder,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            activation_fn=torch.nn.LeakyReLU,
            dropout_rate=dropout_rate,
        )

        self.feature_importance = feature_importance
        self.threshold = threshold
        self.labels = labels

    def inference(self, x, y):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        # x = input_dict['x']
        # y = input_dict['y']
        if self.feature_importance == "random":
            y_mapped = y
        else:
            y_mapped = int_to_str_map(y, self.labels)
        # tbd: adapt for new mask method
        x_masked = mask(x, y_mapped, self.feature_importance, self.threshold)
        qz_m, qz_v, z = self.z_encoder(x_masked)

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v)
        return outputs

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]
        input_dict = dict(x=x, y=y)
        return input_dict


class SCGENCustom(scgen.SCGEN):
    """This class inherits the original SCGEN class and overwrites the initialization.

    This only adds feature importance and threshold to the initialization and passes them to the VAE.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        n_hidden: int = 800,
        n_latent: int = 100,
        n_layers: int = 2,
        dropout_rate: float = 0.2,
        feature_importance=None,  # str,
        threshold=None,  #  float,
        **model_kwargs,
    ):
        super(scgen.SCGEN, self).__init__(adata)

        labels = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)["categorical_mapping"]

        self.module = SCGENVAECustom(
            n_input=self.summary_stats.n_vars,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            feature_importance=feature_importance,
            threshold=threshold,
            labels=labels,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCGEN Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: " "{}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
        )
        self.init_params_ = self._get_init_params(locals())
