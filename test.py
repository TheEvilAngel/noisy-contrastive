import torch
contrast_1 = torch.tensor([
    [0.5000, 0.2000, 0.4100, 0.4600],
    [0.4800, 0.1600, 0.3800, 0.4400],
    [0.4200, 0.2800, 0.3900, 0.4200],
    [0.4400, 0.2400, 0.4200, 0.4000]
])
mask_pos = torch.zeros(4, 4).fill_diagonal_(1)
mask_neg = torch.ones(4, 4).fill_diagonal_(0)
a1 = -contrast_1 * mask_pos # 每个数数乘
a2 = (1 - contrast_1).log() * mask_neg
print(a1)
print(a2)
contrast_1 = -contrast_1 * mask_pos + ((1 - contrast_1).log()) * mask_neg
print(contrast_1)
contrast_logits = 2 + contrast_1
contrast_1 = (-2+1e-4)*torch.zeros(4, 4).fill_diagonal_(1)
print(contrast_1+2)
