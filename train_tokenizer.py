import irishGPT.utilities as uts
import irishGPT.tokenizer as tokenizer

tokenizer = tokenizer.Regex_Tokenizer()
tokenizer.train(uts.get_file_as_string("Datasets/shakespeare.txt"), max_vocab_size=512)

# Save after training
tokenizer.save("shakespeare_tokenizer.json")