from tokenizers import Tokenizer

tok=Tokenizer.from_file("tokenizers_dir/tokenizer_32k.json")

# x=tok.encode("Hello man i play minecraft").ids
# print(x)

print(tok.encode("<BOS>").ids)
print(tok.encode("<EOS>").ids)



