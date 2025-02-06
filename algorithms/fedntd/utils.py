import torch

__all__ = ["refine_as_not_true"]


def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    # print('--------')
    # print('1')
    # print(nt_positions)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    # print('----')
    # print(nt_positions)
    # for each 'sample', list unavailable classes. I.e, if data sample 1 has class 5, list classes 1, 2, 3 ,. ...
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    # print('----')
    # print(nt_positions)
    # print('xx')
    # print(targets)
    nt_positions = nt_positions.view(-1, num_classes - 1)
    # print('----')
    # print(nt_positions)
    # get logits for classes that are not 'true'
    logits = torch.gather(logits, 1, nt_positions)
    # print('----')
    # print(logits)

    return logits
