import torch


def to_diag_var(diag:torch.Tensor):
    diag = diag.squeeze().ravel()
    n = len(diag)
    sigma_mat = torch.eye(n) 
    sigma_mat[torch.arange(n), torch.arange(n)] = diag
    return sigma_mat
