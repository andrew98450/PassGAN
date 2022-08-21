import torchtext

def tokenizer(text):
    chars = []
    text = str(text).replace("\n", "")
    for i in text:
        chars.append(i)
    return chars

def load_dataset(root = "Dataset/train.txt", batch_size = 64, seq_len = 30, vocab_export = False):
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
    
    if vocab_export:
        f = open("Dataset/charmap.txt", "w", encoding="utf-8")
        for i in range(len(TEXT.vocab)):
            f.write("%d,%s\n" % (i, TEXT.vocab.itos[i]))
        f.close()
        
    train_set = torchtext.legacy.data.Iterator(
        dataset=train, 
        batch_size=batch_size,
        shuffle=True)
  
    return train_set, TEXT, len(train), len(TEXT.vocab)

def load_charmap(root = "Dataset/charmap.txt"):
    data = dict()
    with open(root, "r", encoding="utf-8") as f:
        for text in f:
            maps = str(text).split(",", 1)
            data[int(maps[0])] = str(maps[1]).replace("\n", "")
        f.close()
    return data

    