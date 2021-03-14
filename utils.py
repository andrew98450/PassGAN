import torchtext

def tokenizer(text):
    chars = []
    for i in range(len(text)):
        chars.append(str(text[i]))
    return chars

def load_dataset(root = "Dataset/password.txt", batch_size = 64, seq_len = 10):
    TEXT = torchtext.legacy.data.Field(
        unk_token="",
        pad_token="'",
        tokenize=tokenizer, 
        sequential=True,
        use_vocab=True,
        batch_first=True, 
        fix_length=seq_len)   
    train = torchtext.legacy.data.TabularDataset(
        path=root, 
        format="CSV", 
        fields=[("data", TEXT)])
    
    TEXT.build_vocab(train)
    
    train_set = torchtext.legacy.data.Iterator(
        dataset=train, 
        batch_size=batch_size,
        shuffle=True)
    
    return train_set, TEXT, len(train), len(TEXT.vocab)