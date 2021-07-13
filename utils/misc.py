import torch


def to_diag_var(diag:torch.Tensor):
    n = diag.size(0)
    sigma_mat = torch.eye(n) 
    sigma_mat[torch.arange(n), torch.arange(n)] = diag
    return sigma_mat
