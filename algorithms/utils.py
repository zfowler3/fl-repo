import torch
import torch.nn.functional as F

__all__ = [
    "sampled_clients_identifier",
    "tensor_concater",
    "dict_concater",
]


def sampled_clients_identifier(data_distributed, sampled_clients):
    """Identify local datasets information (distribution, size)"""
    local_dist_list, local_size_list = [], []

    for client_idx in sampled_clients:
        local_dist = torch.Tensor(data_distributed["data_map"])[client_idx]
        local_dist = F.normalize(local_dist, dim=0, p=1)
        local_dist_list.append(local_dist.tolist())

        local_size = data_distributed["local"][client_idx]["datasize"]
        local_size_list.append(local_size)

    return local_dist_list, local_size_list


def tensor_concater(tensor1, tensor2, device=None):
    """Concatenate two tensors"""

    if tensor1 is None:
        tensor1 = tensor2

    else:
        if device is not None:
            tensor1 = tensor1.to(device)
            tensor2 = tensor2.to(device)

        tensor1 = torch.cat((tensor1, tensor2), dim=0)

    return tensor1.to(device)


def dict_concater(dict1, dict2):
    """Concatenate two dictionaries"""
    for key, item in dict2.items():
        dict1[key] = item

    return dict1


def deep_leakage_from_gradients(model, origin_data, origin_label, criterion, y):
    dummy_data = torch.randn(origin_data.size())
    dummy_label = torch.randn(origin_label.size())
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    origin_grad = torch.autograd.grad(y, model.parameters())

    for iters in range(300):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = criterion(dummy_pred, F.softmax(dummy_label, dim=-1))
            dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_grad, origin_grad):
                grad_diff += ((gx - gy) ** 2).sum()

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

    return dummy_data, dummy_label
