import torch 

def rmse(y_true, y_pred): 
    return torch.sqrt(torch.mean((y_pred - y_true)**2))

def rel(y_true, y_pred): 
    # calc = torch.nan_to_num(torch.abs(y_pred - y_true) / torch.abs(y_true), nan=0, posinf=0, neginf=0)
    calc = torch.nan_to_num(torch.abs(y_pred - y_true) / y_true, nan=0, posinf=0, neginf=0)
    return torch.mean(calc)

def accuracy_th(y_true, y_pred, threshold): 
    calc1 = torch.nan_to_num(y_pred/y_true, nan=0, posinf=0, neginf=0)
    calc2 = torch.nan_to_num(y_true/y_pred, nan=0, posinf=0, neginf=0)
    calc = torch.maximum(calc1, calc2) 
    return torch.mean( (calc < 1.25**threshold).float() )