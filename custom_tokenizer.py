from tokenizers import BertWordPieceTokenizer


corpus_file = 'D:/project/bert_test/OA_corpus_kor.txt'
vocab_size = 32000
limit_alphabet = 6000

tokenizer = BertWordPieceTokenizer(
    vocab_file=None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False,  # Must be False if cased model
    lowercase=False,
    wordpieces_prefix="##"
)

tokenizer.train(
    files=[corpus_file],
    limit_alphabet=limit_alphabet,
    vocab_size=vocab_size
)

tokenizer.save("./", "oa-{}-wpm-{}".format(limit_alphabet, vocab_size))
