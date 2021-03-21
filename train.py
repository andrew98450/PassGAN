import argparse
import torch
from model import NetG, NetD
from utils import load_dataset

parse = argparse.ArgumentParser()
parse.add_argument("--dataset", default="Dataset/password.txt", type=str)
parse.add_argument("--epoch", default=20, type=int)
parse.add_argument("--gpu", default=False, type=bool)
parse.add_argument("--batchsize", default=500, type=int)
parse.add_argument("--length", default=10, type=int)
parse.add_argument("--lr", default=0.0001, type=float)
args = parse.parse_args()

dataset = args.dataset
seq_len = args.length
batch_size = args.batchsize
lr = args.lr
use_gpu = args.gpu
epoch = args.epoch
c = 0.01
n_c = 5

train_set, text, train_len, vocab_len = load_dataset(
    root=dataset,
    batch_size=batch_size,
    seq_len=seq_len)

netG = NetG(seq_len, vocab_len)
netD = NetD(seq_len, vocab_len)
if use_gpu:
    netG = netG.cuda()
    netD = netD.cuda()

g_optim = torch.optim.Adam(params=netG.parameters(), lr=lr, betas=(0.5, 0.9))
d_optim = torch.optim.Adam(params=netD.parameters(), lr=lr, betas=(0.5, 0.9))

for i in range(epoch):
    for iters, inputs in enumerate(train_set, 0):
        
        real_data = inputs.data.long()
        fake_data = torch.randn(batch_size, 128).float()
        if use_gpu:
            real_data = real_data.cuda()
            fake_data = fake_data.cuda() 
        real_data = torch.nn.functional.one_hot(real_data, vocab_len).float()
        
        for _ in range(n_c):
            generate_data = netG(fake_data).detach()
            d_real = netD(real_data)
            d_fake = netD(generate_data)  
            d_real_loss = -torch.mean(d_real)
            d_fake_loss = torch.mean(d_fake)
            
            d_optim.zero_grad()
            d_real_loss.backward()
            d_fake_loss.backward()
            d_optim.step()
            
            for p in netD.parameters():
                p.data = p.data.clamp(-c, c)
            
        fake_data = torch.randn(batch_size, 128).float()
        if use_gpu:   
            fake_data = fake_data.cuda()             
        generate_data = netG(fake_data)
        d_fake = netD(generate_data)
        g_loss = -torch.mean(d_fake)
        
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()
        
        if iters % 10 == 0:
            print("[+] Epoch: [%d/%d] G_Loss: %.4f D_Real_Loss: %.4f D_Fake_Loss: %.4f" % (i+1, epoch, g_loss, d_real_loss, d_fake_loss))

netG = netG.cpu().eval()
netD = netD.cpu().eval()

torch.save(netG, "pretrained/passwordG_model.pth")  
torch.save(netD, "pretrained/passwordD_model.pth")       