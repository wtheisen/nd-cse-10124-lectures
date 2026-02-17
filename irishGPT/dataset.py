
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from . import utilities as uts

class IrishChatDataset(Dataset):
    def __init__(self, dataset_file, tokenizer):
        self.tokenizer = tokenizer
        self.padding_idx = tokenizer.special_tokens['<|pad|>']
        self.sos_idx = tokenizer.special_tokens['<|sos|>']
        self.eos_idx = tokenizer.special_tokens['<|eos|>']
        self.device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.data = []
        for x in uts.get_file_as_list_strs(dataset_file, special_tokens=True):
            tokens = torch.tensor(self.tokenizer.encode(x), dtype=torch.long)
            if len(tokens) >= 2:
                # X = tokens[:-1], Y = tokens[1:]  (next-token prediction)
                self.data.append((tokens[:-1], tokens[1:]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # ---------- Collate (pad, optional one-hot, move to device) ----------
    def collate(self, batch):
        X_list, Y_list = zip(*batch)
        X = pad_sequence(X_list, batch_first=True, padding_value=self.padding_idx)      # (B,T)
        Y_idx = pad_sequence(Y_list, batch_first=True, padding_value=self.padding_idx)  # (B,T)

        V = len(self.tokenizer.vocab)
        Y = F.one_hot(Y_idx.clamp_min(0), num_classes=V).float()                        # (B,T,V)

        # mask True where this position is a real target token (not padding)
        mask = (Y_idx != self.padding_idx)                                              # (B,T)

        return X.to(self.device), Y.to(self.device), mask.to(self.device)