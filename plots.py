import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import colorcet as cc
import os
import seaborn as sns
import glob

def df_creation_strategies(results_dir, strategies=['fedavg','fedcurv','moon','new'], partition='dirichlet_0.1', sampling_ratio='0.5',
                           dataset='cifar10', n_clients='10', mapping=
                           {'fedavg': '07-15-24','fedcurv': '07-15-24','moon': '07-15-24','new': '07-24-24'},
                           mode='global'):
    '''
    Basic function for comparing multiple FL strategies. Compile into one dataframe + use seaborn to plot all on same
    graph.
    '''
    df = pd.DataFrame([])
    for strategy in strategies:
        print(strategy)
        path = results_dir + mapping[strategy] + '/' + dataset + '/' + partition + '/' + n_clients + '_Clients_' + strategy + '_' + sampling_ratio + '/'
        print(path)
        seeds = glob.glob(path + '*')
        for seed in seeds:
            print(seed)
            new_path = seed + '/'
            if mode == 'global':
                file = new_path + 'global_results.xlsx'
            elif mode == 'global_on_local':
                file = new_path + mode + '.xlsx'
            else:
                file = new_path + 'local_results.xlsx'
            print(file)
            temp = pd.read_excel(file)
            temp['Strategy'] = strategy
            df = pd.concat([df, temp])
    return df

def plot_strategies(results_dir='/home/zoe/Dropbox (GhassanGT)/Zoe/InSync/BIGandDATA/Federated_Learning/',
                    strategies=['fedavg','fedcurv','moon','new'], partition='dirichlet_0.1', sampling_ratio='0.5',
                    dataset='cifar10', n_clients='10', mapping=
                    {'fedavg': '07-15-24', 'fedcurv': '07-15-24', 'moon': '07-15-24', 'new': '07-24-24'}, mode='global',
                    plotting_col='Global test acc'):

    save_path = results_dir + 'Plots/' + dataset + '/' + partition + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = df_creation_strategies(results_dir=results_dir, strategies=strategies, partition=partition, sampling_ratio=sampling_ratio,
                                dataset=dataset, n_clients=n_clients, mapping=mapping, mode=mode)

    # Mapping of plot titles and axis labels based on what you are plotting
    y_labels = {
        'Global test acc': "Test Accuracy on Global Test Set",
        'Global nfr': "NFR on Global Test Set",
        'Backward transfer': 'Backward Transfer',
        'Global model NFR': "Avg NFR on Local Client Test Sets", ##
        'Global model test acc': "Avg Test Accuracy on Local Client Test Sets", ##
        'local on local nfr': 'Avg NFR on Local Test Sets',
        'local on global nfr': 'Avg NFR on Global Test Set',
        'local on local test acc': 'Avg Accuracy on Local Test Sets',
        'local on global test acc': 'Avg Accuracy on Global Test Set'
    }
    titles = {
        'Global test acc': "Global Model Performance on Global Test Set (Acc)",
        'Global nfr': "Global Model Performance on Global Test Set (NFR)",
        'Backward transfer': 'Global Model Performance on Prior Sampled Clients',
        'Global model NFR': "Global Model Performance on Local Test Sets (NFR)",
        'Global model test acc': "Global Model Performance on Local Test Sets (Acc)",
        'local on local nfr': 'Avg Local Model Performance on Local Test Sets (NFR)',
        'local on global nfr': 'Avg Local Model Performance on Global Test Set (NFR)',
        'local on local test acc': 'Avg Local Model Performance on Local Test Sets (Acc)',
        'local on global test acc': 'Avg Local Model Performance on Global Test Set (Acc)'
    }

    save_file = save_path + plotting_col.replace(' ', '_') + '.png'
    y = y_labels[plotting_col]
    title=titles[plotting_col] + ' - ' + dataset

    palette = sns.color_palette(cc.glasbey, n_colors=len(strategies))

    ax = sns.lineplot(data=df, x='Round', y=plotting_col, hue='Strategy', palette=palette)
    plt.legend(loc='upper left', prop={'size': 8})
    plt.grid()
    plt.ylabel(y)
    plt.xlabel('Round')
    plt.title(title, fontsize=11)
    plt.savefig(save_file)
    plt.clf()
    return