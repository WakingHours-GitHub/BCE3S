import numpy as np
import torch
import torch.nn as nn
import math

# The classes below wrap core functions to impose weight regurlarization constraints in training or finetuning a network.
            
class MaxNorm_via_PGD():
    # learning a max-norm constrainted network via projected gradient descent (PGD) 
    def __init__(self, thresh=1.0, LpNorm=2, tau = 1, active_layers=None):
        self.thresh = thresh
        self.LpNorm = LpNorm
        self.tau = tau
        self.perLayerThresh = []
        self.active_layers = active_layers
        
    def setPerLayerThresh(self, model):
        # set per-layer thresholds
        self.perLayerThresh = []
        if not self.active_layers: # is None
            self.active_layers = [model.weight, model.b]
            # self.active_layers = [model.weight]
            
        # for curLayer in [model.encoder.fc.weight, model.encoder.fc.bias]: #here we only apply MaxNorm over the last two layers
        # for curLayer in [model.weight, model.b]: #here we only apply MaxNorm over the last two layers
        for curLayer in self.active_layers: #here we only apply MaxNorm over the last two layers
            curparam = curLayer.data
            if len(curparam.shape)<=1: 
                self.perLayerThresh.append(float('inf'))
                continue
            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1).detach().unsqueeze(-1)
            curLayerThresh = neuronNorm_curparam.min() + self.thresh*(neuronNorm_curparam.max() - neuronNorm_curparam.min())
            self.perLayerThresh.append(curLayerThresh)
                
    def PGD(self, model):
        if len(self.perLayerThresh)==0:
            self.setPerLayerThresh(model)
        
        # for i, curLayer in enumerate([model.encoder.fc.weight, model.encoder.fc.bias]): #here we only apply MaxNorm over the last two layers
        for i, curLayer in enumerate([model.weight, model.b]): #here we only apply MaxNorm over the last two layers
        # for i, curLayer in enumerate(self.active_layers): #here we only apply MaxNorm over the last two layers
        
            curparam = curLayer.data

            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = (torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1)**self.tau).detach().unsqueeze(-1)
            scalingVect = torch.ones_like(curparam)    
            curLayerThresh = self.perLayerThresh[i]
            
            idx = neuronNorm_curparam > curLayerThresh
            idx = idx.squeeze()
            tmp = curLayerThresh / (neuronNorm_curparam[idx].squeeze())**(self.tau)
            for _ in range(len(scalingVect.shape)-1):
                tmp = tmp.unsqueeze(-1)

            scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
            curparam[idx] = scalingVect[idx] * curparam[idx] 

class Normalizer(): 
    def __init__(self, LpNorm=2, tau = 1, active_layers=None):
        self.LpNorm = LpNorm
        self.tau = tau
        self.active_layers = active_layers
        
    def apply_on(self, model, mode="classifier"): #this method applies tau-normalization on the classifier layer
        
        # for curLayer in [model.encoder.fc.weight]: #change to last layer: Done
        # for curLayer in [model.identity.weight]: #change to last layer: Done
        if self.active_layers is None:
            if mode == "classifier": # if model is classifier, we get its weight. 
                weights = [model.weight]
            else:
                weights = [model.encoder.fc.weight]
        else:
            weights = self.active_layers
        
        for curLayer in weights: #change to last layer: Done
            curparam = curLayer.data

            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = (torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1)**self.tau).detach().unsqueeze(-1)
            scalingVect = torch.ones_like(curparam)    
            
            idx = neuronNorm_curparam == neuronNorm_curparam
            idx = idx.squeeze()
            tmp = 1 / (neuronNorm_curparam[idx].squeeze())
            for _ in range(len(scalingVect.shape)-1):
                tmp = tmp.unsqueeze(-1)

            scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
            curparam[idx] = scalingVect[idx] * curparam[idx]

