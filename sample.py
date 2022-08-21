import os
import argparse
import torch
from model import NetG
from utils import load_charmap

parse = argparse.ArgumentParser()
parse.add_argument("--vocabset", default="Dataset/charmap.txt", type=str)
parse.add_argument("--savepath", default="sample/gen_password.txt", type=str)
parse.add_argument("--modelpath", default="pretrained/passwordG_model.pth", type=str)
parse.add_argument("--numsample", default=100000, type=int)
parse.add_argument("--gpu", default=False, type=bool)
parse.add_argument("--batchsize", default=1000, type=int)
parse.add_argument("--length", default=18, type=int)
args = parse.parse_args()

vocabset = args.vocabset
save_path = args.savepath
model_path = args.modelpath
num_sample = args.numsample
batch_size = args.batchsize
seq_len = args.length
use_gpu = args.gpu
current_size = 0

charmap = load_charmap(vocabset)

with open(save_path, "w+") as f:
    while current_size < num_sample:
        inputs = torch.randn(batch_size, 128).float()
        model = torch.load(model_path).eval()
        if use_gpu:
            inputs = inputs.cuda()
            model = model.cuda()
        
        output = model(inputs)
        sample = output.argmax(2).cpu()   
        
        for i in range(batch_size):
            gen_pass = ""
            for j in range(seq_len):   
                index = sample[i][j].item()
                gen_pass += str(charmap[index])
            gen_pass = str(gen_pass).replace("'", "") + "\n"
            f.write(gen_pass)
        current_size += batch_size
        print("[+] Sample Generate Progess: %d %" % int((current_size / num_sample) * 100))           

    f.close()
