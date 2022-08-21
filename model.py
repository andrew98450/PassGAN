import torch

class ResBlock(torch.nn.Module):
 
    def __init__(self, in_channel, out_channel): 
        super(ResBlock, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channel, out_channel, 3, padding=1, bias=False),
            torch.nn.BatchNorm1d(out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(out_channel, out_channel, 3, padding=1, bias=False),
            torch.nn.BatchNorm1d(out_channel))
        
    def forward(self, inputs):
        outputs = self.conv_layer(inputs)
        return (outputs * 0.3) + inputs
    
class NetG(torch.nn.Module):
    
    def __init__(self, seq_len, vocab_len):
        super(NetG, self).__init__()
        self.seq_len = seq_len
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(128, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128 * seq_len, bias=False),
            torch.nn.BatchNorm1d(128 * seq_len),
            torch.nn.ReLU())
        
        self.conv_layer = torch.nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),             
            ResBlock(128, 128),            
            torch.nn.Conv1d(128, vocab_len, 3, padding=1, bias=False),
            torch.nn.Softmax(1))

    def forward(self, inputs):
        outputs = self.fc_layer(inputs)
        outputs = outputs.reshape(-1, 128, self.seq_len)
        outputs = self.conv_layer(outputs)
        outputs = outputs.permute(0, 2, 1)
        return outputs
        
class NetD(torch.nn.Module):
    
    def __init__(self, seq_len, vocab_len):
        super(NetD, self).__init__()
        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(vocab_len, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            ResBlock(128, 128),
            ResBlock(128, 128), 
            ResBlock(128, 128),
            ResBlock(128, 128),             
            ResBlock(128, 128))
        
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * seq_len, 512, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1, bias=False))        
    
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.conv_layer(inputs)
        outputs = self.fc_layer(outputs)
        return outputs    
        
        
