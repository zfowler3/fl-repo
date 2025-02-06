import time
import copy
import os
import sys
import pandas as pd
import torch
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.moon.ClientTrainer import ClientTrainer
from algorithms.moon.criterion import ModelContrastiveLoss
from algorithms.BaseServer import BaseServer
from algorithms.measures import *

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, dataset, save_folder, stats, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, dataset, save_folder, stats, **kwargs
        )
        """
        Server class controls the overall experiment.
        """
        moon_criterion = ModelContrastiveLoss(algo_params.mu, algo_params.tau)

        self.client = ClientTrainer(
            moon_criterion,
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
            save_folder=self.save_folder,
            dataset=self.dataset,
            local_stats=stats
        )

        if self.resume:
            self.prev_locals = np.load(self.save_folder + 'prev_locals.npy', allow_pickle=True)
            self.prev_locals = self.prev_locals.tolist()
        else:
            self.prev_locals = []
            self._init_prev_locals()

        print("\n>>> MOON Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.current_round, self.n_rounds):
            print('Round ' + str(round_idx))

            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            self.client_history[round_idx] = sampled_clients
            self.stats["client_history"] = self.client_history
            self.stats["round"] = round_idx
            # Client training stage to upload weights & stats
            updated_local_weights, client_sizes, round_results = self._clients_training(
                sampled_clients, round_idx
            )

            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)
            # Update global weights and evaluate statistics
            self.df_glob = self._update_and_evaluate(ag_weights, round_results, round_idx, start_time,
                                                     sampled_clients=sampled_clients, df_global=self.df_glob)
            # Save temporary model
            model_path = os.path.join(self.save_folder, "model.pth")
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, model_path)
            # save spreadsheet
            self.df_glob.to_excel(self.save_folder + 'global_results.xlsx', index=False)
            self.df_local.to_csv(self.save_folder + 'global_on_local.csv', index=False)
            self.local_clients_info.to_excel(self.save_folder + 'local_results.xlsx', index=False)

    def _clients_training(self, sampled_clients, r_idx):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = [], []
        round_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:

            # Fetch client datasets
            self._set_client_data(client_idx, noise=r_idx)

            # Download global
            self.client.download_global(
                server_weights, server_optimizer, self.prev_locals[client_idx]
            )

            # Local training
            local_results, local_size = self.client.train()

            df = pd.DataFrame.from_dict([local_results])
            df['Round'] = r_idx
            self.local_clients_info = pd.concat([self.local_clients_info, df])

            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            for local_weights, client in zip(updated_local_weights, sampled_clients):
                self.prev_locals[client] = local_weights

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        np.save(self.save_folder + '/prev_locals.npy', self.prev_locals, allow_pickle=True)

        return updated_local_weights, client_sizes, round_results

    def _init_prev_locals(self):
        weights = self.model.state_dict()
        for _ in range(self.n_clients):
            self.prev_locals.append(copy.deepcopy(weights))
