"""
MIT License

Copyright (c) 2023 Manuel Roeder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import os
import signal
from typing import Dict, Tuple

# torch
import torch

# flower framework
import flwr.client
from flwr.client import start_numpy_client
from flwr.common.typing import NDArrays, Scalar

# project common
from common import signal_handler_free_cuda, parse_client_config, Defaults, check_create_dir

# dassl imports
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger, collect_env_info
from lccs.imcls.train import setup_cfg, print_args

# register lccs-specific trainers and models - import is sufficient
import lccs.imcls.trainers.lccs
from lccs.imcls.trainers.lccs import AbstractLCCS
import lccs.imcls.models.resnet_lccs
# lccs utilities
import lccs.imcls.trainers.lccs_utils.lccs_svd as lccs_utils


"""
If you get an error like: “failed to connect to all addresses” “grpc_status”:14 
Then uncomment the lines bellow:
"""
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]


"""
Capture KB interrupt and free cuda memory
"""
signal.signal(signal.SIGINT, signal_handler_free_cuda)


class FedMixStyleClient(flwr.client.NumPyClient):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__()

        self.trainer: AbstractLCCS = None
        # build trainer upfront
        self.build_trainer()
        # set up data loaders
        self.create_k_shot_loaders()
        self.c_id = cfg.DATASET.NAME + "_" + cfg.DATASET.TARGET_DOMAINS[0]

    def build_trainer(self):
        if not self.trainer:
            self.trainer = build_trainer(self.cfg)

    def load_model(self, model_dir):
        if self.trainer:
            self.trainer.load_model_nostrict(model_dir)

    def create_k_shot_loaders(self):
        if self.trainer:
            self.trainer.get_ksupport_loaders()

    def model_init(self):
        if self.trainer:
            self.trainer.initialization_stage()

    def model_update_gradient(self):
        if self.trainer:
            self.trainer.gradient_update_stage()

    def get_lccs_parameters(self):
        if self.trainer.model_optms is not None:
            optms_params, _ = self.trainer.get_optms_params(component='LCCS')
            return [val.cpu().detach().numpy() for val in optms_params]
        else:
            return list()

    def get_classifier_parameters(self):
        if self.trainer.model_optms is not None:
            optms_params, _ = self.trainer.get_optms_params(component='classifier')
            return [val.cpu().detach().numpy() for val in optms_params]
        else:
            return list()

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : NDArrays
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """
        print("Fit called")
        self.model_update_gradient()
        metrics = dict()
        metrics["c_id"] = self.c_id
        return self.get_lccs_parameters(), 155, metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) have been
        deprecated and removed since Flower 0.19.
        """
        print("Client eval called")
        results = self.trainer.test()
        print(results)
        results.update({"c_id": self.c_id})
        return 0.0, 0, results


def main() -> None:
    args = parse_client_config()

    # setup data source dir
    check_create_dir(args.root)

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    client_model = FedMixStyleClient(cfg=cfg)
    # load pre-trained model
    client_model.load_model(args.model_dir)
    # init model
    client_model.model_init()

    # start flwr client
    try:
        start_numpy_client(server_address=Defaults.SERVER_ADDRESS,
                           client=client_model,
                           grpc_max_message_length=Defaults.GRPC_MAX_MSG_LENGTH)

    except RuntimeError as err:
        print(repr(err))


if __name__ == "__main__":
    # available gpu checks
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        # clear the cache of the current device
        torch.cuda.empty_cache()
        print("[CLIENT] Using CUDA acceleration")
    else:
        print("[CLIENT] Using CPU acceleration")

    # run main
    main()

    # clear cuda cache
    torch.cuda.empty_cache()
    print("[CLIENT] Graceful shutdown")