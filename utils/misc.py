import torch
import numpy as np


def to_diag_var(diag:torch.Tensor,cuda=False):
    diag = diag.squeeze().ravel()
    n = len(diag)
    sigma_mat = torch.eye(n) 
    if cuda:
        sigma_mat = sigma_mat.cuda()
    sigma_mat[torch.arange(n), torch.arange(n)] = diag
    return sigma_mat



def safenumpy(tensor: torch.Tensor) -> np.ndarray:
    if isinstance(tensor,  torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor