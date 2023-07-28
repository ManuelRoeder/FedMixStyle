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

import numpy as np
import torch
# Flwr
import flwr
from flwr.common import Parameters, Scalar, FitRes, \
    EvaluateRes, parameters_to_ndarrays, NDArrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import aggregate
import plot
# dassl.pytorch
from dassl.engine import build_trainer

AGGREGATION_METHOD = "average"# aggregate_median
PLOT_ON_ARRIVAL = True


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
        self.data_tracker = list(dict())
        self.prime_weights = None

    def __repr__(self) -> str:
        return "FedMixStyle"

    def aggregate_centroids(self, round, method):
        # aggregate centroids
        data_dict = self.data_tracker[round - 1]
        aggr_list = list()
        for key in data_dict.keys():
            _, centroids, num_examples = data_dict[key]
            aggr_list.append((centroids, num_examples))
        if method == "average":
            result = aggregate.aggregate(aggr_list)
        else:
            result = aggregate.aggregate_median(aggr_list)
        return result

    def aggregate_weights(self, round, method):
        # aggregate centroids
        data_dict = self.data_tracker[round-1]
        aggr_list = list()

        for key in data_dict.keys():
            weights, _, num_examples = data_dict[key]
            aggr_list.append((weights, num_examples))
        if method == "average":
            result = aggregate.aggregate(aggr_list)
        else:
            result = aggregate.aggregate_median(aggr_list)
        return result



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
        tmp_dict = dict()
        for (client, fit_res) in results:
            client_id = str(fit_res.metrics["c_id"])
            lccs_params_size = int(fit_res.metrics["lccs_params_size"])
            centroid_size = int(fit_res.metrics["centroid_size"])
            parameters = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            # split
            lccs_params, centroids = np.split(parameters, [lccs_params_size], axis=0)
            if len(lccs_params) == lccs_params_size and len(centroids) == centroid_size:
                print("Parameter check success")
            # bookmark
            tmp_dict[client_id] = (lccs_params, centroids, num_examples)
            # bookkeeping
            #parameter_count = 0
            #for params in parameters:
                #parameter_count = parameter_count + params.size
            #parameter_size = parameters[0].itemsize
            #total_memory_size = parameter_count * parameter_size
            #print("[STRATEGY] Parameter count: " + str(parameter_count))
            #print("[STRATEGY] Total Memory Transferred in Bytes: " + str(total_memory_size))
            status = fit_res.status
            # send current weights to dict
            print("[STRATEGY] Client " + str(client_id) + " returned result message= " + status.message)

        self.data_tracker.append(tmp_dict)

        # aggregations
        self.weights_prime = self.aggregate_weights(server_round, AGGREGATION_METHOD)
        self.centroids_prime = self.aggregate_centroids(server_round, AGGREGATION_METHOD)

        if PLOT_ON_ARRIVAL:
            centroid_list = list()
            # append mean centroids first
            centroid_list.extend(self.centroids_prime)
            last_element = self.data_tracker[-1]
            for key in last_element.keys():
                _, centroids, _ = last_element[key]
                centroid_list.extend(centroids)
            plot.plot_centroids(centroid_list, len(self.centroids_prime), True)

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
