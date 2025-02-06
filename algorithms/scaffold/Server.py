import pandas as pd
import torch
import copy
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.scaffold.ClientTrainer import ClientTrainer
from algorithms.scaffold.utils import *
from algorithms.measures import *

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, dataset, save_folder, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, dataset, save_folder, **kwargs
        )
        """
        Server class controls the overall experiment.
        """
        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
            dataset=self.dataset
        )

        self.c, self.local_c = self._init_control_variates()

        print("\n>>> Scaffold Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.n_rounds):
            print('Round ' + str(round_idx))

            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            self.client_history[round_idx] = sampled_clients
            #self.server_results["client_history"].append(sampled_clients)

            # Client training stage to upload weights & stats
            (
                updated_local_weights,
                updated_c_amounts,
                client_sizes,
                round_results,
            ) = self._clients_training(sampled_clients, round_idx)

            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)

            # Get aggregated control variates
            with torch.no_grad():
                updated_c_amounts = torch.stack(updated_c_amounts).mean(dim=0)
                self.c += self.sample_ratio * updated_c_amounts

            # Update global weights and evaluate statistics
            self.df_glob = self._update_and_evaluate(ag_weights, round_results, round_idx, start_time,
                                                     sampled_clients=sampled_clients, df_global=self.df_glob)
            # Save temporary model
            model_path = os.path.join(self.save_folder, "model.pth")
            torch.save(self.model.state_dict(), model_path)
        # save spreadsheet
        self.df_glob.to_excel(self.save_folder + 'global_results.xlsx', index=False)
        self.df_local.to_excel(self.save_folder + 'global_on_local.xlsx', index=False)
        self.local_clients_info.to_excel(self.save_folder + 'local_results.xlsx', index=False)

    def _clients_training(self, sampled_clients, r_idx):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, updated_c_amounts, client_sizes = [], [], []
        round_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(
                server_weights, server_optimizer, self.c, self.local_c[client_idx]
            )

            # Local training
            local_results, local_size, c_i_plus, c_update_amount = self.client.train()

            df = pd.DataFrame.from_dict([local_results])
            df['Round'] = r_idx
            self.local_clients_info = pd.concat([self.local_clients_info, df])

            # Upload locals
            updated_local_weights.append(self.client.upload_local())
            updated_c_amounts.append(c_update_amount)
            self.local_c[client_idx] = c_i_plus

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        return updated_local_weights, updated_c_amounts, client_sizes, round_results

    def _init_control_variates(self):
        c = flatten_weights(self.model)
        c = torch.from_numpy(c).fill_(0)

        local_c = []

        for _ in range(self.n_clients):
            local_c.append(copy.deepcopy(c))

        return c, local_c
