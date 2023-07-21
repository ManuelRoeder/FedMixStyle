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
import torch

# Flower framework
import flwr as fl
from flwr.server import start_server, ServerConfig, SimpleClientManager
from flwr.common import ndarrays_to_parameters

import common
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger, collect_env_info
from lccs.imcls.train import setup_cfg, print_args
from strategy import FedMixStyleStrategy
from common import signal_handler_free_cuda, Defaults


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
os.environ["GRPC_VERBOSITY"] = "debug"

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


class FedMixStyleServer(fl.server.Server):
    @staticmethod
    def add_server_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FedMixStyleServer")
        parser.add_argument("--host_address", type=str, default=Defaults.SERVER_ADDRESS)
        parser.add_argument("--num_rounds", type=int, default=1)
        parser.add_argument("--max_msg_size", type=int, default=Defaults.GRPC_MAX_MSG_LENGTH)
        #parser.add_argument("--server_side_evaluation", default=False, type=boolean_string)
        #parser.add_argument("--force_final_distributed_eval", default=False, type=boolean_string)
        return parent_parser

    def __init__(self, strategy):
        # create the client manager
        client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)


def main() -> None:
    args = common.parse_server_config()

    # setup data source dir
    common.check_create_dir(args.root)

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

    trainer = build_trainer(cfg)

    # Dassl.pytorch/dassl/engine/trainer.py
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    # try to load model
    trainer.load_model(args.model_dir)
    if not trainer._models:
        if not args.no_train:
            # pretrain the server model
            trainer.train()
        else:
            return

    # STRATEGY CONFIGURATION: pass pretrained model to server
    NUM_CLIENTS = 1
    fms_strategy = FedMixStyleStrategy(
        model_dir=args.model_dir,
        config=cfg,
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=NUM_CLIENTS)

    # SERVER SETUP
    server = FedMixStyleServer(strategy=fms_strategy)

    # Server config
    server_config = ServerConfig(num_rounds=1, round_timeout=99999)

    try:
        # Start Lightning Flower server for three rounds of federated learning
        start_server(server=server,
                     server_address=Defaults.SERVER_ADDRESS,
                     config=server_config,
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
        print("[SERVER] Using CUDA acceleration")
    else:
        print("[SERVER] Using CPU acceleration")

    # configure precision
    torch.set_float32_matmul_precision('highest')

    # start and run FL server
    main()

    # clear cuda cache
    torch.cuda.empty_cache()
    print("[SERVER] Graceful shutdown")
