import torch
from torch import optim, nn
from tqdm import trange
import argparse
from dataloader import get_loader
from model import MDD
# from torch.utils.tensorboard import SummaryWriter

def main():

    # load data
    train_loader = get_loader(s='mnist', t='mnist_m')
    test_loader = get_loader(s='mnist', t='mnist_m', train=False)
    # writer = SummaryWriter(log_dir=f'./runs/mdd')

    # init model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MDD(base_net='ResNet50', width=1024, use_gpu=True, class_num=10, srcweight=1.5)

    # init optimizer
    opt = optim.SGD(model.get_parameter_list(), lr=5e-3, weight_decay=5e-5, momentum=0.9, nesterov=True)

    def train(e):
        # model.train()
        cls_loss_avg = 0
        trs_loss_avg = 0
        with trange(len(train_loader)) as t:
            for i, (src, tgt) in enumerate(train_loader):
                simg, slabel, sdomain = src
                timg, tlabel, tdomain = tgt
                img = torch.cat([simg, timg], dim=0)
                label = torch.cat([slabel, tlabel], dim=0)
                domain = torch.cat([sdomain, tdomain], dim=0)
                img, label, domain = img.to(device), label.to(device), domain.to(device)
                
                alpha = torch.tensor(2.0 * 0.1 / (1. + torch.exp(torch.tensor(-(i+e*50)/1000.))) - 0.1)


                cls_loss, trans_loss = model.get_loss(img, label, domain, alpha=alpha)
                cls_loss_avg += cls_loss.item()
                trs_loss_avg += trans_loss.item()
                loss = cls_loss + trans_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

                metric = {'cls': cls_loss_avg/(i+1),
                          'trs': trs_loss_avg/(i+1)}
                t.set_postfix(metric)
                t.update()
        # writer.add_scalar('cls_los', cls_loss_avg/(i+1), e)
        # writer.add_scalar('trans_loss', trs_loss_avg/(i+1), e)
    
    def test(e):
        total = 0
        correct = 0
        with torch.no_grad():
            with trange(len(test_loader)) as t:
                for i, (timg, tlabel, tdomain) in enumerate(test_loader):
                    img, label = timg.to(device), tlabel.to(device)
                    pred = model.predict(img)

                    pred = pred.argmax(dim=1)
                    correct += pred.eq(label).sum().item()
                    total += len(label)

                    metric = {'acc': correct/total}
                    t.set_postfix(metric)
                    t.update()
            # writer.add_scalar('acc', correct/total, e)
    for e in range(200):
        print('==='*10, e, '==='*10)
        train(e)
        test(e)

if __name__ == "__main__":
    main()