import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class Class_Balanced_Loss_with_normalize_weight(nn.Module):
    def __init__(self, in_features, out_features, img_num_per_cls:list, mode, beta, gamma, r=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.img_num_per_cls = img_num_per_cls
        self.nClasses = out_features
        self.mode = mode
        self.beta = beta
        self.gamma = gamma
        self.r = r
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features)) # 类中心。
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        
        self.b = nn.Parameter(torch.FloatTensor(out_features))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)
    
    
    def focal_loss(self, labels, logits, alpha, gamma):
        """Compute the focal loss between `logits` and the ground truth `labels`.

        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

        Args:
        labels: A float tensor of size [batch, num_classes].
        logits: A float tensor of size [batch, num_classes].
        alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
        gamma: A float scalar modulating loss from hard and easy examples.

        Returns:
        focal_loss: A float32 scalar representing normalized total loss.
        """    
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
                torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss

    
    def forward(self, feature, labels, epoch=None, weight_old=None):
        # feature[labels.sort()[1], :].norm(dim=1) feature norm
        # self.weight.norm(dim=1) 调试信息
        
        # print(self.weight.norm(dim=1))
        logits = F.linear(feature, F.normalize(self.weight, eps=1e-5), self.b)
        
        if weight_old is None:
            effective_num = 1.0 - np.power(self.beta, self.img_num_per_cls)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * self.nClasses
        else:
            weights = weight_old

        labels_one_hot = F.one_hot(labels.long(), self.nClasses).float()
        labels_one_hot = labels_one_hot.to(labels)

        weights = torch.tensor(weights).float().cuda()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1)
        weights = weights * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.nClasses)

        if self.mode == "focal":
            cb_loss = self.focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.mode == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
        elif self.mode == "softmax":
            pred = logits.softmax(dim = 1)
            labels_one_hot = labels_one_hot.float()
            # cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
            cb_loss = F.binary_cross_entropy_with_logits(input = pred, target = labels_one_hot, weight = weights)
        elif self.mode == 'bce':
            positive = torch.unique(labels, sorted=True).long().to(device = labels.device)
            perm = torch.randperm(self.out_features, device = labels.device)
            perm[positive] = 0
            indices = torch.topk(perm, k = int(self.out_features * self.r), largest = False)[1]
            partial_index = indices.sort()[0]

            partial_logits = logits[:, partial_index]
            
            one_hot = torch.zeros((labels.size(0), logits.size(-1)), dtype=torch.bool)
            one_hot = one_hot.to(logits.device)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            one_hot = torch.index_select(one_hot, 1, partial_index)
            weights = torch.index_select(weights, 1, partial_index)
            
            p_loss = torch.log(1 + torch.exp(-partial_logits))
            n_loss = torch.log(1 + torch.exp( partial_logits))
            loss = (one_hot * p_loss + (~one_hot) * n_loss)
            cb_loss = (loss * weights).sum(dim=1).mean()
            
        return cb_loss, logits
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"img_num_per_cls={self.img_num_per_cls}, "
                f"nClasses={self.nClasses}, "
                f"mode={self.mode}, "
                f"beta={self.beta}, "
                f"gamma={self.gamma}, "
                f"r={self.r})")
        
        
        

class ContrastiveLearning(nn.Module):
    """
    Handles sample-to-sample contrastive learning logic.
    """
    def __init__(self, in_features, projection_dim, out_features, temperature=0.1, strict=64, eps=1e-5):
        super().__init__()
        self.in_features = in_features
        self.projection_dim = projection_dim
        self.out_features = out_features
        self.temperature = temperature
        self.strict = strict
        self.eps = eps
        self.register_buffer('feat_memory', torch.FloatTensor(out_features, projection_dim))
        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, projection_dim)
        )

    def update_memory(self, input, label):
        label = label.long()
        for i in range(label.size(0)):
            self.feat_memory[label[i]] = input[i].data.float()

    def forward(self, features, labels, onehot, neg_mask):
        projected = self.projector(features)
        logits = F.linear(F.normalize(projected, eps=self.eps), F.normalize(self.feat_memory, eps=self.eps)) / self.temperature
        return logits

class UniformLearning(nn.Module):
    """
    Handles classifier-to-classifier similarity logic.
    """
    def __init__(self, out_features, cc_s=1.0, cc_weight=0.1, eps=1e-5):
        super().__init__()
        self.out_features = out_features
        self.cc_s = cc_s
        self.cc_weight = cc_weight
        self.eps = eps
        self.cc_one_hot = torch.eye(self.out_features, dtype=torch.bool)

    def forward(self, weight):
        self.cc_one_hot = self.cc_one_hot.cuda() if weight.is_cuda else self.cc_one_hot
        class_class_sim = self.cc_s * F.linear(F.normalize(weight, eps=self.eps), F.normalize(weight, eps=self.eps))
        pos_class_loss = torch.log(1 + torch.exp(-class_class_sim))
        neg_class_loss = torch.log(1 + torch.exp(class_class_sim))
        cc_loss = self.cc_one_hot * pos_class_loss + (~self.cc_one_hot) * neg_class_loss
        cc_loss = cc_loss.sum(dim=1).mean()
        return cc_loss

# Main Tripartite Synergistic Learning Module
class BCE_TripartiteSynergisticLearning(nn.Module):
    """
    Combines sample-to-sample and classifier-to-classifier modules for tripartite synergistic learning.
    """
    def __init__(self, in_features, out_features, projection_dim, strict=64, r=1.0, l=1.0, temperature=0.1, ss_weight=0.1, cc_s=1.0, cc_weight=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.strict = strict
        self.r = r
        self.l = l
        self.ss_weight = ss_weight
        self.cc_s = cc_s
        self.cc_weight = cc_weight
        self.eps = 1e-5

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.b = nn.Parameter(torch.FloatTensor(out_features))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

        self.sample_module = ContrastiveLearning(in_features, projection_dim, out_features, temperature, strict, self.eps)
        self.classifier_module = UniformLearning(out_features, cc_s, cc_weight, self.eps)

    def bce_loss(self, logits, one_hot, neg_mask=None):
        logits = logits.clamp(-self.strict, self.strict)
        p_loss = torch.log(1 + torch.exp(-logits))
        n_loss = torch.log(1 + torch.exp(logits)) * self.l
        if neg_mask is not None:
            loss = (one_hot * p_loss + neg_mask * n_loss)
        else:
            loss = (one_hot * p_loss + (~one_hot) * n_loss)
        loss = loss.sum(dim=1).mean()
        return loss

    def update_memory(self, input, label):
        self.sample_module.update_memory(input, label)

    def forward(self, inputs, origin_label, mixup_onehot=None, cutmix_onehot=None, epoch=None, phase='test'):
        if phase == 'train':
            # Partial index selection
            positive = torch.unique(origin_label, sorted=True).long().to(device=origin_label.device)
            perm = torch.randperm(self.out_features, device=origin_label.device)
            perm[positive] = 0
            indices = torch.topk(perm, k=int(self.out_features * self.r), largest=False)[1]
            partial_index = indices.sort()[0]

            bs = origin_label.size(0)
            origin_feature, mixup_feature, cutmix_feature = torch.split(inputs, [bs, bs, bs])

            logits = F.linear(origin_feature, F.normalize(self.weight, eps=1e-5), bias=self.b)
            partial_logits = logits[:, partial_index]

            one_hot = torch.zeros((origin_label.size(0), self.out_features), dtype=torch.bool)
            one_hot = one_hot.cuda() if logits.is_cuda else one_hot
            one_hot.scatter_(1, origin_label.view(-1, 1).long(), 1)
            one_hot = torch.index_select(one_hot, 1, partial_index)

            origin_loss = self.bce_loss(partial_logits, one_hot)
            origin_p = self.sample_module.projector(origin_feature)
            neg_mask = (mixup_onehot == 0).float()

            mixup_f_logits = F.linear(mixup_feature, F.normalize(self.weight, eps=self.eps), bias=self.b)
            cutmix_f_logits = F.linear(cutmix_feature, F.normalize(self.weight, eps=self.eps), bias=self.b)
            mixup_f_loss = self.bce_loss(mixup_f_logits, mixup_onehot, neg_mask)
            cutmix_f_loss = self.bce_loss(cutmix_f_logits, mixup_onehot, neg_mask)
            sc_loss = (origin_loss + mixup_f_loss + cutmix_f_loss) / 3

            # Classifier-to-classifier loss
            cc_loss = self.classifier_module(self.weight)

            if (epoch is not None) and epoch == 0:
                self.update_memory(origin_p, origin_label)
                return sc_loss + self.cc_weight * cc_loss, logits

            # Sample-to-sample loss
            mixup_p = self.sample_module.projector(mixup_feature)
            cutmix_p = self.sample_module.projector(cutmix_feature)
            mixup_logits = self.sample_module(mixup_feature, origin_label, mixup_onehot, neg_mask)
            cutmix_logits = self.sample_module(cutmix_feature, origin_label, mixup_onehot, neg_mask)
            mixup_loss = self.bce_loss(mixup_logits, mixup_onehot, neg_mask)
            cutmix_loss = self.bce_loss(cutmix_logits, mixup_onehot, neg_mask)
            ss_loss = mixup_loss + cutmix_loss

            self.update_memory(origin_p, origin_label)
            return sc_loss + self.ss_weight * ss_loss + self.cc_weight * cc_loss, logits
        else:
            logits = F.linear(inputs, F.normalize(self.weight, eps=1e-5), bias=self.b)
            one_hot = torch.zeros((origin_label.size(0), self.out_features), dtype=torch.bool)
            one_hot = one_hot.cuda() if logits.is_cuda else one_hot
            one_hot.scatter_(1, origin_label.view(-1, 1).long(), 1)
            bce_loss = self.bce_loss(logits, one_hot)
            return bce_loss, logits

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"temperature={self.temperature}, "
                f"projection_dim={self.projection_dim}, "
                f"strict={self.strict}, "
                f"r={self.r}, "
                f"l={self.l}, "
                f"ss_weight={self.ss_weight}, "
                f"cc_s={self.cc_s}, "
                f"cc_weight={self.cc_weight}, "
                f"eps={self.eps})")
