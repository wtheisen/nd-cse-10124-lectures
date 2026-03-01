import irishGPT.utilities as uts
import irishGPT.tokenizer as tokenizer

tokenizer = tokenizer.Regex_Tokenizer()
tokenizer.train(uts.get_file_as_string("Datasets/openweb10k.txt"), max_vocab_size=3200)

# Save after training
tokenizer.save("openweb10k_tokenizer.json")