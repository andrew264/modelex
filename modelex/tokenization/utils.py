from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers, processors

def create_byte_level_tokenizer(path: str, special_tokens: tuple[str] = ('<s>', '</s>', '<pad>')):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=256+len(special_tokens),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=list(special_tokens),)
    data = ['hello, world!']
    tokenizer.train_from_iterator(data, trainer=trainer)
    tokenizer.save(path + '/tokenizer.json')
