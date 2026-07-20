from tokenizer import train_bpe

text = open(r"C:\Users\apkar\OneDrive\Documents\Plivo test\llm_handout\llm_handout\data\train_corpus.txt", encoding="utf-8").read()
tok = tok = train_bpe(text, target_vocab_size=1500)
tok.save()
assert tok.decode(tok.encode(text[:5000])) == text[:5000]
print("vocab_size:", tok.vocab_size)