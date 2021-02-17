import torch
from torch.nn import CrossEntropyLoss
from tqdm import trange

from pointnet2.models.point_transformer_part_seg import PointTransformerPartSeg


def main():
    print('==> Testing Part Segmentation Network...')
    B, N = 8, 5000
    in_channels = 3
    num_classes = 10

    net = PointTransformerPartSeg(in_channels, num_classes).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss()

    pbar = trange(10000)
    for i in pbar:
        optimizer.zero_grad()
        x = torch.randn(B, N, in_channels).cuda()
        p = torch.randn(B, N, 3).cuda()
        target = torch.zeros((B*N,), dtype=torch.long).cuda()
        y, out_p = net(x, p)
        loss = crit(y.view(-1, num_classes), target)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'Loss': loss.item()})

        if i % 10 == 0:
            torch.cuda.empty_cache()

        if i == 0:
            assert torch.all(torch.eq(p, out_p)), "point permuted!"
            print('[Inputs] x, p')
            print('    x.shape:', x.shape)
            print('    p.shape:', p.shape)
            print('[Outputs] y')
            print('    y.shape:', y.shape)


if __name__ == '__main__':
    main()
