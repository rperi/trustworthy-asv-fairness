import torch
from sklearn.metrics import accuracy_score

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_spk_accuracy(true_lab, pred_lab):
    return accuracy_score(true_lab, pred_lab)