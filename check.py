import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.special_tokens_set)