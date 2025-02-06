import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.fedprox.ClientTrainer import ClientTrainer

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
            self, algo_params, model, data_distributed, optimizer, scheduler, dataset,
            save_folder, stats, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, dataset,
            save_folder, stats, **kwargs
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
        print("\n>>> FedProx Server initialized...\n")
