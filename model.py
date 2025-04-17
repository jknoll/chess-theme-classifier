# Model adapted from the hackathon-3-winner architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Maps to: white king, queen, rook, bishop, knight, pawn, empty, black pawn, knight, bishop, rook, queen, king
PIECE_CHARS = "♔♕♖♗♘♙⭘♟♞♝♜♛♚"  

# Dataset uses 0-12 indices for chess pieces

# Lamb optimizer implementation from chess-hackathon
class Lamb(torch.optim.Optimizer):
    """
    Implements Lamb algorithm from 'Large Batch Optimization for Deep Learning: Training BERT in 76 minutes'
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0,
        clamp_value=10,
        adam=False,
        debias=False,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if weight_decay < 0:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state['step'])
                    bias_correction /= 1 - beta1 ** state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss

# Learning rate scheduler with warmup
def get_lr_with_warmup(step, total_steps, warmup_steps, base_lr):
    """
    Learning rate scheduler with warmup from chess-hackathon
    Args:
        step: Current step
        total_steps: Total steps in the epoch
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate
    Returns:
        Learning rate with warmup applied
    """
    lr_factor = min(step / warmup_steps, 1.0)
    return lr_factor * base_lr

class Attention(nn.Module):
    '''
    Implements a temporal attention block with a provision to increase the number of
    heads to two
    '''
    def __init__(self, input_dims, attention_dims, n_heads=2):
        super().__init__()
        self.attention_dims = attention_dims
        self.n_heads = n_heads
        self.k1 = nn.Linear(input_dims, attention_dims)
        self.v1 = nn.Linear(input_dims, attention_dims)
        self.q1 = nn.Linear(input_dims, attention_dims)
        
        if n_heads == 2:
            self.k2 = nn.Linear(input_dims, attention_dims)
            self.v2 = nn.Linear(input_dims, attention_dims)
            self.q2 = nn.Linear(input_dims, attention_dims)
            self.attention_head_projection = nn.Linear(attention_dims * 2, input_dims)
        else:
            self.attention_head_projection = nn.Linear(attention_dims, input_dims)

        self.activation = nn.Softmax(dim=-1)
        
    def forward(self, x):
        '''
        x: shape (B,D,k1,k2) where B is the Batch size, D is number of filters, and k1, k2 are the kernel sizes
        '''
        oB, oD, oW, oH = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(oB, -1, oD)

        q1, v1, k1 = self.q1(x), self.v1(x), self.k1(x)
        qk1 = (q1@k1.permute((0,2,1)))/(self.attention_dims ** 0.5)
        multihead = self.activation(qk1)@v1 
        if self.n_heads == 2:
            q2, v2, k2 = self.q2(x), self.v2(x), self.k2(x)
            qk2 = (q2@k2.permute((0,2,1)))/(self.attention_dims ** 0.5) 
            attention = self.activation(qk2)@v2       
            multihead = torch.cat((multihead, attention), dim=-1)
   
        multihead_concat = self.attention_head_projection(multihead)     # shape: (B, 64, 64)
        return multihead_concat.reshape(oB, oD, oW, oH)

class Residual(nn.Module):
    """
    The Residual block of ResNet models.
    """
    def __init__(self, outer_channels, inner_channels, use_1x1conv, dropout_rate, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(outer_channels, inner_channels, kernel_size=3, padding='same', stride=1, dilation=dilation)
        self.conv2 = nn.Conv2d(inner_channels, outer_channels, kernel_size=3, padding='same', stride=1, dilation=dilation)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(outer_channels, outer_channels, kernel_size=1, stride=1)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.bn2 = nn.BatchNorm2d(outer_channels)
        
        # Handle both float and nn.Dropout
        if isinstance(dropout_rate, float):
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = dropout_rate

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.dropout(self.bn2(self.conv2(Y)))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class Model(nn.Module):
    """
    Modified version of the winning model that maintains multi-label classification
    """
    def __init__(self, num_labels=62, nlayers=2, embed_dim=64, inner_dim=320, 
                 attention_dim=64, use_1x1conv=True, dropout=0.5):
        super().__init__()
        self.vocab = PIECE_CHARS
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim
        self.use_1x1conv = use_1x1conv
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.num_labels = num_labels

        # Use embedding size of 13 to match dataset piece indices (0-12)
        self.embedder = nn.Embedding(13, self.embed_dim)
        self.convLayers = nn.ModuleList()
        for i in range(nlayers): 
            self.convLayers.append(Residual(self.embed_dim, self.inner_dim, self.use_1x1conv, self.dropout, 2**i))
            self.convLayers.append(Attention(self.embed_dim, attention_dim))

        self.convnet = nn.Sequential(*self.convLayers)
        self.accumulator = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=8, padding=0, stride=1)
        
        # Multi-label output layers (keeping the fully connected layers from original model)
        self.fc1 = nn.Linear(self.embed_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_labels)  # Output layer for multi-label classification
        
        self.init_weights()
        
        # Print model parameters count
        self.print_model_size()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedder.weight, -initrange, initrange)
        nn.init.uniform_(self.fc3.weight, -initrange, initrange)
        
    def print_model_size(self):
        """Calculate and print the number of parameters in the model"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params_count = sum([torch.prod(torch.tensor(p.size())).item() for p in model_parameters])
        print(f"Model initialized with {params_count:,} trainable parameters")

    def forward(self, input, debug=False):
        # Input reshaping for embedder - handle the input format from the dataset
        if input.dim() == 4 and input.size(1) == 1:  # (batch, 1, height, width)
            # Convert from float to long for embedding
            input = input.squeeze(1).long()
        
        if debug: print(f"Input shape before embedding: {input.shape}, dtype: {input.dtype}")
        
        # Embedding expects indices between 0 and len(PIECE_CHARS)-1
        # Our dataset uses 0-12 which maps to chess pieces, need to ensure compatibility
        x = self.embedder(input)
        if debug: print(f"After direct embedding: {x.shape}")
        
        x = torch.permute(x, (0, 3, 1, 2)).contiguous()
        if debug: print(f"After permute: {x.shape}")
        
        x = self.convnet(x)
        if debug: print(f"After convnet: {x.shape}")
        
        x = self.accumulator(x)
        if debug: print(f"After accumulator (before squeeze): {x.shape}")
        
        x = F.relu(x.squeeze())
        if debug: print(f"After accumulator + relu: {x.shape}")
        
        # Fully connected layers with dropout for multi-label classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        if debug: print(f"After fc1: {x.shape}")
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        if debug: print(f"After fc2: {x.shape}")
        
        x = self.fc3(x)
        if debug: print(f"After fc3: {x.shape}")
        
        return x