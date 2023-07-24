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
    EvaluateRes, parameters_to_ndarrays, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# dassl.pytorch
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
        # track adaptation history
        self.adaptation_history = Dict[str, NDArrays]

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
            client_id = str(fit_res.metrics["c_id"])
            #duration = fit_res.metrics["duration"]
            #classifier_loss = fit_res.metrics["classifier_loss"]
            parameters = parameters_to_ndarrays(fit_res.parameters)
            parameter_count = 0
            for params in parameters:
                parameter_count = parameter_count + params.size
            parameter_size = parameters[0].itemsize
            total_memory_size = parameter_count * parameter_size
            print("[STRATEGY] Parameter count: " + str(parameter_count))
            #print("[STRATEGY] Parameter size: " + str(parameter_size))
            print("[STRATEGY] Total Memory Transferred in Bytes: " + str(total_memory_size))

            status = fit_res.status
            # send current weights to dict
            self.adaptation_history.update({client_id: parameters})
            print("[STRATEGY] Client " + str(client_id) + " returned result message= " + status.message)

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
            client_id = str(eval_res.metrics["c_id"])
            #duration = eval_res.metrics["duration"]
            accuracy = eval_res.metrics["accuracy"]
            error_rate = eval_res.metrics["error_rate"]
            macro_f1 = eval_res.metrics["macro_f1"]
            status = eval_res.status
            print("[STRATEGY] Client " + str(client_id) + " returned result message= " + status.message + " with accuracy=" + str(accuracy) + ", error rate=" + str(error_rate) + ", f1_score=" + str(macro_f1))
        return None, {}
