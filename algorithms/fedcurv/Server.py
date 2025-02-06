import numpy as np
import pandas as pd
import torch
import copy
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.measures import *
from algorithms.fedcurv.ClientTrainer import ClientTrainer
from algorithms.BaseServer import BaseServer

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
        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
            save_folder=self.save_folder,
            dataset=self.dataset,
            local_stats=stats
        )

        # Dictionaries for saving 'diag(Hess)'(=ut) and 'diag(Hess)*local_weight'(=vt)
        if self.resume:
            self.updated_local_uts = np.load(self.save_folder + 'local_uts.npy', allow_pickle=True).item(0)
            self.updated_local_vts = np.load(self.save_folder + 'local_vts.npy', allow_pickle=True).item(0)
        else:
            self.updated_local_uts = {}
            self.updated_local_vts = {}

        print("\n>>> FedCurv Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.current_round, self.n_rounds):

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
            model_path = os.path.join(self.save_folder, "model.pth")
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, model_path)
            # save spreadsheet
            self.df_glob.to_excel(self.save_folder + 'global_results.xlsx', index=False)
            self.df_local.to_csv(self.save_folder + 'global_on_local.csv', index=False)
            self.local_clients_info.to_excel(self.save_folder + 'local_results.xlsx', index=False)

    def _clients_training(self, sampled_clients, round_idx):
        """
        Conduct local training and get trained local models' weights
        Now _clients_training function takes round_idx
        (Since we can not use Fisher regularization on the very first round; round_idx=0)
        """

        updated_local_weights, client_sizes = [], []
        round_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        temp_fisher = None
        temp_theta_fisher = None

        # Unless the round > 0, we don't have Fisher regularizer
        if round_idx != 0:
            # Get global Ut and Vt
            with torch.no_grad():
                Ut = torch.sum(
                    torch.stack(list(self.updated_local_uts.values())), dim=0
                )
                Vt = torch.sum(
                    torch.stack(list(self.updated_local_vts.values())), dim=0
                )

        # Client training stage
        for client_idx in sampled_clients:

            # Fetch client datasets
            self._set_client_data(client_idx, noise=round_idx)

            # Download global
            self.client.download_global(server_weights, server_optimizer)

            # Download Fisher regularizer
            if round_idx != 0:
                with torch.no_grad():
                    Pt = Ut
                    Qt = Vt

                    if client_idx in self.updated_local_vts:
                        Pt -= self.updated_local_uts[client_idx]
                        Qt -= self.updated_local_vts[client_idx]

                self.client.download_fisher_regularizer(Pt, Qt)

            # Local training
            local_results, local_size = self.client.train()
            #print('Local Results: ', local_results)
            # save local results
            df = pd.DataFrame.from_dict([local_results])
            df['Round'] = round_idx
            self.local_clients_info = pd.concat([self.local_clients_info, df])

            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            # Upload 'diag(Hess)'(=ut) and 'diag(Hess) dot optimized weight'(=vt)
            # Uploaded vector is stored at the dictionary, having client_idx as the key
            local_ut, local_vt = self.client.upload_local_fisher()
            self.updated_local_uts[client_idx] = local_ut
            self.updated_local_vts[client_idx] = local_vt

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        np.save(self.save_folder + 'local_uts.npy', self.updated_local_uts, allow_pickle=True)
        np.save(self.save_folder + 'local_vts.npy', self.updated_local_vts, allow_pickle=True)

        return updated_local_weights, client_sizes, round_results
