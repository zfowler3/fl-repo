import json
import os
import ast

__all__ = ["ConfLoader", "directory_setter", "config_overwriter"]


class ConfLoader:
    """
    Load json config file using DictWithAttributeAccess object_hook.
    ConfLoader(conf_name).opt attribute is the result of loading json config file.
    """

    class DictWithAttributeAccess(dict):
        """
        This inner class makes dict to be accessed same as class attribute.
        For example, you can use opt.key instead of the opt['key']
        """

        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    def __init__(self, conf_name):
        self.conf_name = conf_name
        self.opt = self.__get_opt()

    def __load_conf(self):
        with open(self.conf_name, "r") as conf:
            opt = json.load(
                conf, object_hook=lambda dict: self.DictWithAttributeAccess(dict)
            )
        return opt

    def __get_opt(self):
        opt = self.__load_conf()
        opt = self.DictWithAttributeAccess(opt)

        return opt


def directory_setter(path="./results", make_dir=False):
    """
    Make dictionary if not exists.
    """
    if not os.path.exists(path) and make_dir:
        os.makedirs(path)  # make dir if not exist
        print("directory %s is created" % path)

    if not os.path.isdir(path):
        raise NotADirectoryError(
            "%s is not valid. set make_dir=True to make dir." % path
        )


def config_overwriter(opt, args):
    """
    Overwrite loaded configuration by parsing arguments.
    """
    if args.dataset_name is not None:
        opt.data_setups.dataset_name = args.dataset_name

    if args.date is not None:
        opt.data_setups.date = args.date

    if args.spreadsheet_path is not None:
        opt.spreadsheet_path = args.spreadsheet_path

    if args.root is not None:
        opt.data_setups.root = args.root

    if args.base_folder is not None:
        opt.data_setups.base_folder = args.base_folder

    if args.batch_size is not None:
        opt.data_setups.batch_size = args.batch_size

    if args.n_clients is not None:
        opt.data_setups.n_clients = args.n_clients

    if args.partition_method is not None:
        opt.data_setups.partition.method = args.partition_method

    if args.partition_alpha is not None:
        opt.data_setups.partition.alpha = args.partition_alpha

    if args.model_name is not None:
        opt.train_setups.model.name = args.model_name

    if args.n_rounds is not None:
        opt.train_setups.scenario.n_rounds = args.n_rounds

    if args.sample_ratio is not None:
        opt.train_setups.scenario.sample_ratio = ast.literal_eval(str(args.sample_ratio))

    if args.local_epochs is not None:
        opt.train_setups.scenario.local_epochs = args.local_epochs

    if args.device is not None:
        opt.train_setups.scenario.device = args.device

    if args.lr is not None:
        opt.train_setups.optimizer.params.lr = args.lr

    if args.momentum is not None:
        opt.train_setups.optimizer.params.momentum = args.momentum

    if args.wd is not None:
        opt.train_setups.optimizer.params.weight_decay = args.wd

    if args.algo_name is not None:
        opt.train_setups.algo.name = args.algo_name

    if args.seed is not None:
        opt.train_setups.seed = args.seed

    if args.seed is not None:
        opt.seed = args.seed

    if args.resume is not None:
        opt.resume = args.resume

    if args.shift is not None:
        opt.data_setups.local_setups.shift_type = args.shift

    if args.beta is not None:
        opt.train_setups.algo.params.beta = args.beta

    if args.multilabel is not None:
        opt.train_setups.algo.params.multilabel = args.multilabel

    if args.continual is not None:
        opt.train_setups.algo.params.continual = args.continual

    opt.class_comp = args.class_comp

    return opt
