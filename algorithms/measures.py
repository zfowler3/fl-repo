import numpy as np
import torch
import torch.nn.functional as F
from .utils import *
from sklearn.metrics import f1_score

__all__ = ["evaluate_model", "evaluate_model_classwise", "get_round_personalized_acc"]


@torch.no_grad()
def evaluate_model(model, dataloader, dataset, device="cuda:0"):
    """Evaluate model accuracy for the given dataloader"""
    model.eval()
    model.to(device)

    running_count = 0
    running_correct = 0
    out_list = []
    label_list = []

    for data, targets, _ in dataloader:

        data, targets = data.to(device), targets.to(device)
        logits = model(data)

        if dataset == 'olives':
            labels = targets.float() # biomarker tensor
            if (labels.squeeze().detach().cpu().numpy()).ndim < 2:
                l = np.atleast_2d(labels.squeeze().detach().cpu().numpy())
            else:
                l = labels.squeeze().detach().cpu().numpy()
            label_list.append(l)
            output = torch.round(torch.sigmoid(logits)) # Gets class labels
            #print(output)
            if (output.squeeze().detach().cpu().numpy()).ndim < 2:
                o = np.atleast_2d(output.squeeze().detach().cpu().numpy())
            else:
                o = output.squeeze().detach().cpu().numpy()
            out_list.append(o)
        else:
            pred = logits.max(dim=1)[1]
            running_correct += (targets == pred).sum().item()
            running_count += data.size(0)

    if dataset == 'olives':
        # Get macro-averaged f1 score for biomarker detection
        accuracy = {}
        #print(out_list)
        accuracy['macro'] = f1_score(np.concatenate(label_list), np.concatenate(out_list), average='macro')
        accuracy['class'] = f1_score(np.concatenate(label_list), np.concatenate(out_list), average='macro')
    else:
        accuracy = round(running_correct / running_count, 4)

    return accuracy

@torch.no_grad()
def evaluate_model_split(model, dataloader, dataset, device="cuda:0"):
    """Evaluate model accuracy for the given dataloader"""
    model.eval()
    model.to(device)

    running_count = 0
    running_correct = 0
    out_list = []
    label_list = []

    for data, targets, _ in dataloader:

        data, targets = data.to(device), targets.to(device)
        rep = model.base(data)
        logits = model.head(rep)

        if dataset == 'olives':
            labels = targets.float() # biomarker tensor
            if (labels.squeeze().detach().cpu().numpy()).ndim < 2:
                l = np.atleast_2d(labels.squeeze().detach().cpu().numpy())
            else:
                l = labels.squeeze().detach().cpu().numpy()
            label_list.append(l)
            output = torch.round(torch.sigmoid(logits)) # Gets class labels
            #print(output)
            if (output.squeeze().detach().cpu().numpy()).ndim < 2:
                o = np.atleast_2d(output.squeeze().detach().cpu().numpy())
            else:
                o = output.squeeze().detach().cpu().numpy()
            out_list.append(o)
        else:
            pred = logits.max(dim=1)[1]
            running_correct += (targets == pred).sum().item()
            running_count += data.size(0)

    if dataset == 'olives':
        # Get macro-averaged f1 score for biomarker detection
        accuracy = {}
        #print(out_list)
        accuracy['macro'] = f1_score(np.concatenate(label_list), np.concatenate(out_list), average='macro')
        accuracy['class'] = f1_score(np.concatenate(label_list), np.concatenate(out_list), average='macro')
    else:
        accuracy = round(running_correct / running_count, 4)

    return accuracy

@torch.no_grad()
def evaluate_model_DBE(model, dataloader, dataset, client_mean, device="cuda:0"):
    """Evaluate model accuracy for the given dataloader"""
    model.eval()
    model.to(device)

    running_count = 0
    running_correct = 0
    out_list = []
    label_list = []

    for data, targets, _ in dataloader:
        data, targets = data.to(device), targets.to(device)
        rep = model.base(data)
        logits = model.head(rep + client_mean)

        if dataset == 'olives':
            labels = targets.float() # biomarker tensor
            if (labels.squeeze().detach().cpu().numpy()).ndim < 2:
                l = np.atleast_2d(labels.squeeze().detach().cpu().numpy())
            else:
                l = labels.squeeze().detach().cpu().numpy()
            label_list.append(l)
            output = torch.round(torch.sigmoid(logits)) # Gets class labels
            if (output.squeeze().detach().cpu().numpy()).ndim < 2:
                o = np.atleast_2d(output.squeeze().detach().cpu().numpy())
            else:
                o = output.squeeze().detach().cpu().numpy()
            out_list.append(o)
        else:
            pred = logits.max(dim=1)[1]
            running_correct += (targets == pred).sum().item()
            running_count += data.size(0)

    if dataset == 'olives':
        # Get macro-averaged f1 score for biomarker detection
        accuracy = {}
        accuracy['macro'] = f1_score(np.concatenate(label_list), np.concatenate(out_list), average='macro')
        accuracy['class'] = f1_score(np.concatenate(label_list), np.concatenate(out_list), average='macro')
    else:
        accuracy = round(running_correct / running_count, 4)

    return accuracy


@torch.no_grad()
def model_metrics(model, dataloader, previous_acc, dataset, device="cuda:0"):
    """Evaluate NFR for the given dataloader"""
    model.eval()
    model.to(device)

    forgets = 0
    running_count = 0
    running_correct = 0
    amount = 0
    gt = []
    predictions = []

    for data, targets, idx in dataloader:
        if dataset == 'olives':
            targets = targets.float()
        data, targets = data.to(device), targets.to(device)
        if (targets.squeeze().detach().cpu().numpy()).ndim < 2:
            t = np.atleast_2d(targets.squeeze().detach().cpu().numpy())
        else:
            t = targets.squeeze().detach().cpu().numpy()
        gt.append(t)
        logits = model(data)

        if dataset == 'olives':
            pred = torch.round(torch.sigmoid(logits)) # convert to multi-label vector
        else:
            pred = logits.max(dim=1)[1]

        if (pred.squeeze().detach().cpu().numpy()).ndim < 2:
            p = np.atleast_2d(pred.squeeze().detach().cpu().numpy())
        else:
            p = pred.squeeze().detach().cpu().numpy()
        predictions.append(p)

        accuracy = pred.eq(targets.data)
        delta = np.clip(previous_acc[idx] - accuracy.cpu().numpy(), a_min=0, a_max=1)
        forgets += np.sum(delta)
        previous_acc[idx] = accuracy.cpu().numpy() # Tensor of False/True if samples matched 'targets' or not
        #
        if dataset == 'olives':
            amount += (data.size(0)*len(targets[0])) # Number of patients * number of options they can have (5 in the case of olives)
        else:
            running_correct += (targets == pred).sum().item()
            running_count += data.size(0)
            amount += data.size(0)

    # Return previous acc, forgets, and NFR
    # 'Accuracy' score changes for multilabel
    if dataset == 'olives':
        acc = {}
        # Report F1 score for multilabel
        acc['macro'] = f1_score(y_true=np.concatenate(gt), y_pred=np.concatenate(predictions), average='macro')
        acc['class'] = f1_score(y_true=np.concatenate(gt), y_pred=np.concatenate(predictions), average=None)
    else:
        acc = round(running_correct / running_count, 4)

    return previous_acc, forgets, forgets/amount, acc

@torch.no_grad()
def model_metrics_glob(model, dataloader, previous_acc, dataset, device="cuda:0"):
    """Evaluate NFR for the given dataloader"""
    model.eval()
    model.to(device)

    forgets = 0
    positive_flips = 0
    unchanged = 0
    running_count = 0
    running_correct = 0
    amount = 0
    gt = []
    predictions = []

    for data, targets, idx in dataloader:
        #print('targets shape: ', targets.shape)
        if dataset == 'olives':
            targets = targets.float()
        data, targets = data.to(device), targets.to(device)
        if (targets.squeeze().detach().cpu().numpy()).ndim < 2:
            t = np.atleast_2d(targets.squeeze().detach().cpu().numpy())
        else:
            t = targets.squeeze().detach().cpu().numpy()
        gt.append(t)
        logits = model(data)

        if dataset == 'olives':
            pred = torch.round(torch.sigmoid(logits)) # convert to multi-label vector
        else:
            pred = logits.max(dim=1)[1]

        if (pred.squeeze().detach().cpu().numpy()).ndim < 2:
            p = np.atleast_2d(pred.squeeze().detach().cpu().numpy())
        else:
            p = pred.squeeze().detach().cpu().numpy()
        predictions.append(p)

        accuracy = pred.eq(targets.data)
        delta = np.clip(previous_acc[idx] - accuracy.cpu().numpy(), a_min=0, a_max=1)
        positive_delta = np.clip(accuracy.cpu().numpy() - previous_acc[idx], a_min=0, a_max=1)
        forgets += np.sum(delta)
        positive_flips += np.sum(positive_delta)
        tot_flips = np.sum(delta) + np.sum(positive_delta)
        previous_acc[idx] = accuracy.cpu().numpy() # Tensor of False/True if samples matched 'targets' or not
        #
        if dataset == 'olives':
            amount += (data.size(0)*len(targets[0])) # Number of patients * number of options they can have (5 in the case of olives)
            unchanged += (data.size(0)*len(targets[0]) - tot_flips)
        else:
            running_correct += (targets == pred).sum().item()
            running_count += data.size(0)
            amount += data.size(0)
            unchanged += (data.size(0) - tot_flips)

    # Return previous acc, forgets, and NFR
    # 'Accuracy' score changes for multilabel
    if dataset == 'olives':
        acc = {}
        # Report F1 score for multilabel
        acc['macro'] = f1_score(y_true=np.concatenate(gt), y_pred=np.concatenate(predictions), average='macro')
        acc['class'] = f1_score(y_true=np.concatenate(gt), y_pred=np.concatenate(predictions), average=None)
    else:
        acc = round(running_correct / running_count, 4)

    # print('pos flips: ', positive_flips)
    # print('negative: ', forgets)
    # print('unchanged:', unchanged)

    return previous_acc, forgets, forgets/amount, acc, positive_flips, unchanged

@torch.no_grad()
def model_metrics_split(model, dataloader, previous_acc, dataset, device="cuda:0"):
    """Evaluate NFR for the given dataloader"""
    model.eval()
    model.to(device)

    forgets = 0
    running_count = 0
    running_correct = 0
    amount = 0
    gt = []
    predictions = []

    for data, targets, idx in dataloader:
        if dataset == 'olives':
            targets = targets.float()
        data, targets = data.to(device), targets.to(device)
        if (targets.squeeze().detach().cpu().numpy()).ndim < 2:
            t = np.atleast_2d(targets.squeeze().detach().cpu().numpy())
        else:
            t = targets.squeeze().detach().cpu().numpy()
        gt.append(t)
        rep = model.base(data)
        logits = model.head(rep)

        if dataset == 'olives':
            pred = torch.round(torch.sigmoid(logits)) # convert to multi-label vector
        else:
            pred = logits.max(dim=1)[1]

        if (pred.squeeze().detach().cpu().numpy()).ndim < 2:
            p = np.atleast_2d(pred.squeeze().detach().cpu().numpy())
        else:
            p = pred.squeeze().detach().cpu().numpy()
        predictions.append(p)

        accuracy = pred.eq(targets.data)
        delta = np.clip(previous_acc[idx] - accuracy.cpu().numpy(), a_min=0, a_max=1)
        forgets += np.sum(delta)
        previous_acc[idx] = accuracy.cpu().numpy() # Tensor of False/True if samples matched 'targets' or not
        #
        if dataset == 'olives':
            amount += (data.size(0)*len(targets[0])) # Number of patients * number of options they can have (5 in the case of olives)
        else:
            running_correct += (targets == pred).sum().item()
            running_count += data.size(0)
            amount += data.size(0)

    # Return previous acc, forgets, and NFR
    # 'Accuracy' score changes for multilabel
    if dataset == 'olives':
        acc = {}
        # Report F1 score for multilabel
        acc['macro'] = f1_score(y_true=np.concatenate(gt), y_pred=np.concatenate(predictions), average='macro')
        acc['class'] = f1_score(y_true=np.concatenate(gt), y_pred=np.concatenate(predictions), average=None)
    else:
        acc = round(running_correct / running_count, 4)

    return previous_acc, forgets, forgets/amount, acc

@torch.no_grad()
def model_metrics_DBE(model, dataloader, previous_acc, dataset, client_mean, device="cuda:0"):
    """Evaluate NFR for the given dataloader"""
    model.eval()
    model.to(device)

    forgets = 0
    running_count = 0
    running_correct = 0
    amount = 0
    gt = []
    predictions = []

    for data, targets, idx in dataloader:
        data, targets = data.to(device), targets.to(device)
        if (targets.squeeze().detach().cpu().numpy()).ndim < 2:
            t = np.atleast_2d(targets.squeeze().detach().cpu().numpy())
        else:
            t = targets.squeeze().detach().cpu().numpy()
        gt.append(t)

        rep = model.base(data)
        logits = model.head(rep + client_mean)

        if dataset == 'olives':
            pred = torch.round(torch.sigmoid(logits)) # convert to multi-label vector
        else:
            pred = logits.max(dim=1)[1]

        if (pred.squeeze().detach().cpu().numpy()).ndim < 2:
            p = np.atleast_2d(pred.squeeze().detach().cpu().numpy())
        else:
            p = pred.squeeze().detach().cpu().numpy()
        predictions.append(p)

        accuracy = pred.eq(targets.data)
        delta = np.clip(previous_acc[idx] - accuracy.cpu().numpy(), a_min=0, a_max=1)
        forgets += np.sum(delta)
        previous_acc[idx] = accuracy.cpu().numpy() # Tensor of False/True if samples matched 'targets' or not
        #
        if dataset == 'olives':
            amount += (data.size(0)*len(targets[0])) # Number of patients * number of options they can have (5 in the case of olives)
        else:
            running_correct += (targets == pred).sum().item()
            running_count += data.size(0)
            amount += data.size(0)

    # Return previous acc, forgets, and NFR
    # 'Accuracy' score changes for multilabel
    if dataset == 'olives':
        acc = {}
        # Report F1 score for multilabel
        acc['macro'] = f1_score(y_true=np.concatenate(gt), y_pred=np.concatenate(predictions), average='macro')
        acc['class'] = f1_score(y_true=np.concatenate(gt), y_pred=np.concatenate(predictions), average=None)
    else:
        acc = round(running_correct / running_count, 4)

    return previous_acc, forgets, forgets/amount, acc


@torch.no_grad()
def evaluate_model_classwise(
    model, dataloader, num_classes=10, device="cuda:0",
):
    """Evaluate class-wise accuracy for the given dataloader."""

    model.eval()
    model.to(device)

    classwise_count = torch.Tensor([0 for _ in range(num_classes)]).to(device)
    classwise_correct = torch.Tensor([0 for _ in range(num_classes)]).to(device)

    for data, targets, idx in dataloader:

        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        preds = logits.max(dim=1)[1]

        for class_idx in range(num_classes):
            class_elem = targets == class_idx
            classwise_count[class_idx] += class_elem.sum().item()
            classwise_correct[class_idx] += (targets == preds)[class_elem].sum().item()

    idxs = np.where(classwise_count.cpu().numpy() != 0)
    classwise_accuracy = classwise_correct[idxs] / classwise_count[idxs]
    accuracy = round(classwise_accuracy.mean().item(), 4)

    full_classwise = np.zeros(shape=num_classes)
    for j in range(num_classes):
        try:
            full_classwise[j] = classwise_correct[j].item()/classwise_count[j].item()
            #print(classwise_correct[j].item()/classwise_count[j].item())
        except ZeroDivisionError:
            full_classwise[j] = 0

    return classwise_accuracy.cpu(), accuracy, full_classwise

@torch.no_grad()
def evaluate_model_classwise_split(
    model, dataloader, num_classes=10, device="cuda:0",
):
    """Evaluate class-wise accuracy for the given dataloader."""

    model.eval()
    model.to(device)

    classwise_count = torch.Tensor([0 for _ in range(num_classes)]).to(device)
    classwise_correct = torch.Tensor([0 for _ in range(num_classes)]).to(device)

    for data, targets, idx in dataloader:

        data, targets = data.to(device), targets.to(device)
        rep = model.base(data)
        logits = model.head(rep)
        preds = logits.max(dim=1)[1]

        for class_idx in range(num_classes):
            class_elem = targets == class_idx
            classwise_count[class_idx] += class_elem.sum().item()
            classwise_correct[class_idx] += (targets == preds)[class_elem].sum().item()
    idxs = np.where(classwise_count.cpu().numpy() != 0)
    classwise_accuracy = classwise_correct[idxs] / classwise_count[idxs]
    accuracy = round(classwise_accuracy.mean().item(), 4)

    return classwise_accuracy.cpu(), accuracy

def evaluate_model_classwise_DBE(
    model, dataloader, client_mean, num_classes=10, device="cuda:0",
):
    """Evaluate class-wise accuracy for the given dataloader."""

    model.eval()
    model.to(device)

    classwise_count = torch.Tensor([0 for _ in range(num_classes)]).to(device)
    classwise_correct = torch.Tensor([0 for _ in range(num_classes)]).to(device)

    for data, targets, idx in dataloader:
        data, targets = data.to(device), targets.to(device)

        rep = model.base(data)
        logits = model.head(rep + client_mean)

        preds = logits.max(dim=1)[1]

        for class_idx in range(num_classes):
            class_elem = targets == class_idx
            classwise_count[class_idx] += class_elem.sum().item()
            classwise_correct[class_idx] += (targets == preds)[class_elem].sum().item()

    classwise_accuracy = classwise_correct / classwise_count
    accuracy = round(classwise_accuracy.mean().item(), 4)

    return classwise_accuracy.cpu(), accuracy


@torch.no_grad()
def get_round_personalized_acc(round_results, server_results, data_distributed):
    """Evaluate personalized FL performance on the sampled clients."""

    sampled_clients = server_results["client_history"][-1]
    #print('sampled clients: ', sampled_clients)
    local_dist_list, local_size_list = sampled_clients_identifier(
        data_distributed, sampled_clients
    )
    # print('local size list: ', local_size_list)
    # print('local dist list: ', local_dist_list)

    local_cwa_list = round_results["classwise_accuracy"]
    #print('local classwise acc list: ', local_cwa_list)

    result_dict = {}

    in_dist_acc_list, out_dist_acc_list = [], []
    local_size_prop = F.normalize(torch.Tensor(local_size_list), dim=0, p=1)
    print('local size norm: ', local_size_prop)

    for local_cwa, local_dist_vec in zip(local_cwa_list, local_dist_list):
        # print('loc cwa: ', local_cwa)
        # print('loc dist vec: ', local_dist_vec)
        local_dist_vec = torch.Tensor(local_dist_vec)
        inverse_dist_vec = calculate_inverse_dist(local_dist_vec)
        in_dist_acc = torch.dot(local_cwa, local_dist_vec)
        in_dist_acc_list.append(in_dist_acc)
        out_dist_acc = torch.dot(local_cwa, inverse_dist_vec)
        out_dist_acc_list.append(out_dist_acc)

    round_in_dist_acc = torch.Tensor(in_dist_acc_list)
    round_out_dist_acc = torch.Tensor(out_dist_acc_list)

    result_dict["in_dist_acc_prop"] = torch.dot(
        round_in_dist_acc, local_size_prop
    ).item()
    result_dict["in_dist_acc_mean"] = (round_in_dist_acc).mean().item()
    result_dict["in_dist_acc_std"] = (round_in_dist_acc).std().item()
    result_dict["out_dist_acc"] = torch.dot(round_out_dist_acc, local_size_prop).item()
    result_dict["in_dout_acc_mean"] = (round_out_dist_acc).mean().item()

    return result_dict


@torch.no_grad()
def calculate_inverse_dist(dist_vec):
    """Get the out-local distribution"""
    inverse_dist_vec = (1 - dist_vec) / (dist_vec.nelement() - 1)

    return inverse_dist_vec
