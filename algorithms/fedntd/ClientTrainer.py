import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, criterion, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        self.criterion = criterion

    def train(self):
        """Local training"""

        # Keep global model's weights
        self._keep_global()

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize

        ce_items = []
        ntd_items = []

        for _ in range(self.local_epochs):
            for data, targets, _ in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                if self.dataset == 'olives':
                    targets = targets.float()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                logits, dg_logits = self.model(data), self._get_dg_logits(data)
                loss, ce, ntd = self.criterion(logits, targets, dg_logits)

                ce_items.append(ce)
                ntd_items.append(ntd)

                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats(current_client=self.current_client)

        local_results["kl value"] = np.mean(np.array(ntd_items))
        local_results["cross entropy value"] = np.mean(np.array(ce_items))

        return local_results, local_size

    def _get_dg_logits(self, data):
        with torch.no_grad():
            dg_logits = self.dg_model(data)

        return dg_logits
