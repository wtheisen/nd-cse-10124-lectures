import torch
from irishGPT.irishChat import IrishChat
import irishGPT.utilities as uts
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset
import irishGPT.tokenizer as tokenizer

class IrishChatDataset(Dataset):
    def __init__(self, training_file):
        self.tokenizer = tokenizer.Regex_Tokenizer()

        self.tokenizer.load('Datasets/openweb10k_tokenizer.json')
        # self.tokenizer.train(uts.get_file_as_string(training_file), 512)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.data = []
        for x in uts.get_file_as_list_strs(training_file, special_tokens=True)[:1000]:
            tokens = torch.tensor(self.tokenizer.encode(x), dtype=torch.long)
            if len(tokens) >= 2:
                # X = tokens[:-1], Y = tokens[1:]  (next-token prediction)
                self.data.append((tokens[:-1], tokens[1:]))
        self.padding_idx = 256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # ---------- Collate (pad, optional one-hot, move to device) ----------
    def collate(self, batch):
        X_list, Y_list = zip(*batch)  # tuples of 1D tensors
        X = pad_sequence(X_list, batch_first=True, padding_value=self.padding_idx)        # (B,T)
        Y_idx = pad_sequence(Y_list, batch_first=True, padding_value=self.padding_idx)    # (B,T)

        Y = F.one_hot(Y_idx.clamp_min(0), num_classes=len(self.tokenizer.vocab)).float()        # (B,T,V)
        return X.to(self.device), Y.to(self.device)

if __name__ == "__main__":
    chat = IrishChat(vocab_size=3200)
    dataset = IrishChatDataset("Datasets/openweb10k.txt")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate)
    chat.train(train_loader, 500, 0.01, verbose=True)

    # Interactive chat loop
    print("\n" + "=" * 50)
    print("IrishGPT ready! Type a prompt and press Enter.")
    print("Commands: /temp <value> to change temperature, /quit to exit")
    print("=" * 50 + "\n")

    temperature = 0.8
    while True:
        try:
            prompt = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt.strip():
            continue
        if prompt.strip() == "/quit":
            print("Goodbye!")
            break
        if prompt.strip().startswith("/temp"):
            try:
                temperature = float(prompt.strip().split()[1])
                print(f"  Temperature set to {temperature}")
            except (IndexError, ValueError):
                print(f"  Current temperature: {temperature}")
            continue

        prompt_tokens = dataset.tokenizer.encode("<|sos|>" + prompt + "<|eos|>")[:-1]
        output_tokens = chat.chat(prompt_tokens, max_new_tokens=200, temperature=temperature)
        print(f"IrishGPT: {dataset.tokenizer.decode(output_tokens)}\n")
