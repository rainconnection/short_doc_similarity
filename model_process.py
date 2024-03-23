import torch
import torchvision


def get_loss(infered, label, loss_f, opt = None):
    loss_func = loss_f
        
    loss = loss_func(infered, label)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), len(label) 
