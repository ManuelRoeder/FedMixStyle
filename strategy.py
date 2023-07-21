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
from typing import Optional, Dict, Tuple, List, Union, Any

# Flwr
from flwr.common import Parameters, Scalar, FitRes, \
    EvaluateRes  # , weights_to_parameters, parameters_to_weights, FitIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from dassl.engine import build_trainer


class FedMixStyleStrategy(FedAvg):
    """Configurable FedMixStyle strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        model_dir: str,
        config: Any,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        # build up model and trainer
        self.model_dir = model_dir
        self.config = config
        self.trainer = build_trainer(config)
        self.trainer.load_model(model_dir)

    def __repr__(self) -> str:
        return "FedMixStyle"

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using aggregation strategy from upstreaming"""
        print("[STRATEGY] Aggregate_fit called")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        print("[STRATEGY] Server collecting parameters from round " + str(server_round) + " from number of clients=" + str(len(results)))

        for (client, fit_res) in results:
            client_id = fit_res.metrics["client_id"]
            duration = fit_res.metrics["duration"]
            classifier_loss = fit_res.metrics["classifier_loss"]
            status = fit_res.status
            print("[STRATEGY] Client " + str(client_id) + " returned result message= " + status.message + " with duration " + str(duration) + " and classifier_loss=" + str(classifier_loss))

        return None, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        print("[STRATEGY] Server evaluate on query set in round " + str(
            server_round) + " from number of clients=" + str(len(results)))

        for (client, eval_res) in results:
            client_id = eval_res.metrics["client_id"]
            duration = eval_res.metrics["duration"]
            accuracy = eval_res.metrics["mean_accuracy"]
            deviation = eval_res.metrics["st_dev"]
            status = eval_res.status
            print("[STRATEGY] Client " + str(client_id) + " returned result message= " + status.message + " with duration " + str(duration) + " and mean eval accuracy=" + accuracy + " with deviation=" + str(deviation))
        return None, {}
