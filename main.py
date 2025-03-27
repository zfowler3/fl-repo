import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import algorithms
from train_tools import *
from utils import *
import numpy as np
import argparse
import warnings
import random
import pprint
import os

warnings.filterwarnings("ignore")

# Set torch base print precision
torch.set_printoptions(10)

ALGO = {
    "fedavg": algorithms.fedavg.Server,
    "fedcurv": algorithms.fedcurv.Server,
    "fedprox": algorithms.fedprox.Server,
    "fedavgm": algorithms.fedavgm.Server,
    "fedntd": algorithms.fedntd.Server,
    "fednova": algorithms.fednova.Server,
    "scaffold": algorithms.scaffold.Server,
    "moon": algorithms.moon.Server,
    "feddyn": algorithms.feddyn.Server,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}


def _get_setups(args):

    np.random.seed(args.seed)
    random.seed(args.seed)

    # For data distribution shift
    shift_type = args.data_setups.local_setups["shift_type"]
    extra = ''

    # Create save folder
    base_folder = args.data_setups.base_folder
    folder = str(args.data_setups.n_clients) + '_' + 'Clients_' + args.train_setups.algo.name + '_' + str(args.train_setups.scenario.sample_ratio) + extra

    # Set up folder to save stuff
    if args.data_setups.partition.method == 'dirichlet':
        train_test_folder = (base_folder + '/' + args.data_setups.dataset_name + '/' + \
                      args.data_setups.partition.method + '_' + str(args.data_setups.partition.alpha) + '/' +
                             str(args.data_setups.n_clients) + '_' + 'Clients_Idxs' + extra + '/')

        save_folder = (base_folder + args.data_setups.date + '/' + args.data_setups.dataset_name + '/' +
                       args.data_setups.partition.method + '_' + str(args.data_setups.partition.alpha) + extra + '/' + folder + '/' +
                       str(args.seed) + '/')
    else:
        train_test_folder = base_folder + '/' + args.data_setups.dataset_name + '/' + \
                      args.data_setups.partition.method + '/' + str(args.data_setups.n_clients) + '_' + 'Clients_Idxs' + extra + '/'
        save_folder = (base_folder + args.data_setups.date + '/' + args.data_setups.dataset_name + '/' +
                       args.data_setups.partition.method + '/' + folder + '/' + str(args.seed) + '/')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not os.path.exists(train_test_folder):
        os.makedirs(train_test_folder)

    # Distribute the data to clients
    # Returns dictionary with global and local loaders
    data_distributed = data_distributer(args, root=args.data_setups.root, dataset_name=args.data_setups.dataset_name,
                                        batch_size=args.data_setups.batch_size, n_clients=args.data_setups.n_clients,
                                        partition=args.data_setups.partition, save_folder=train_test_folder,
                                        shift_type=shift_type)

    _random_seeder(args.seed)

    model = create_models(
        args.train_setups.model.name,
        args.data_setups.dataset_name,
        **args.train_setups.model.params,
    )
    # Optimization setups
    optimizer = optim.SGD(model.parameters(), **args.train_setups.optimizer.params)
    scheduler = None

    if args.resume:
        checkpoint = torch.load(save_folder + 'model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )

    dataset = args.data_setups.dataset_name

    algo_params = args.train_setups.algo.params

    if args.resume:
        print()
        print('>>> Resuming from Previous Checkpoint')
        stats = True
    else:
        stats = False

    server = ALGO[args.train_setups.algo.name](
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        dataset,
        save_folder,
        stats,
        **args.train_setups.scenario,
    )

    return server, save_folder


def _random_seeder(seed):
    """Fix randomness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """Execute experiment"""
    # Load the configuration
    server, path = _get_setups(args)

    # Conduct FL
    server.run()

    # Save the final global model
    model_path = os.path.join(path, "model.pth")
    torch.save(server.model.state_dict(), model_path)



######################################################################3

# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Process Configs")
parser.add_argument("--root_path", default="/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/PhDResearch/Code/fl-repo/", type=str)
parser.add_argument("--date", type=str)
parser.add_argument("--shift", type=str)
parser.add_argument("--resume", action='store_true')
parser.add_argument("--continual", action='store_true')
parser.add_argument("--multilabel", action='store_true')
parser.add_argument("--root", default="./data", type=str)
parser.add_argument("--class_comp", type=int)
parser.add_argument("--base_folder", default="/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Federated_Learning/", type=str)
parser.add_argument("--config_path", default="/config/fedavg.json", type=str)
parser.add_argument("--spreadsheet_path", default="/home/zoe/spreadsheets", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--n_clients", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--partition_method", type=str)
parser.add_argument("--partition_s", type=int)
parser.add_argument("--partition_alpha", type=float)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_rounds", type=int)
parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--local_epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--algo_name", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--beta", type=float)
args = parser.parse_args()

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.root_path + args.config_path).opt

    # Overwrite config by parsed arguments
    opt = config_overwriter(opt, args)

    # Print configuration dictionary pretty
    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)

    # Execute experiment
    main(opt)