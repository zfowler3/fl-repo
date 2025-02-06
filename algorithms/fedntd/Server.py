import torch
import torch.nn as nn
import numpy as np
import copy
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.fedntd.ClientTrainer import ClientTrainer
from algorithms.fedntd.criterion import *

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, dataset, save_folder, stats, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, dataset, save_folder, stats, **kwargs
        )
        local_criterion = self._get_local_criterion(self.algo_params, self.num_classes)

        self.client = ClientTrainer(
            local_criterion,
            algo_params=self.algo_params,
            model=copy.deepcopy(self.model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
            save_folder=self.save_folder,
            dataset=self.dataset,
            local_stats=stats
        )

        print("\n>>> FedNTD Server initialized...\n")

    def _get_local_criterion(self, algo_params, num_classes):
        tau = algo_params.tau
        beta = algo_params.beta

        criterion = NTD_Loss(num_classes, tau, beta)

        return criterion
