from transformers import GPT2Tokenizer, GPT2LMHeadModel
import tensorflow as tf

tf.random.set_seed(0)
tokenizer = GPT2Tokenizer.from_pretrained('output')
model = GPT2LMHeadModel.from_pretrained('output')
ids = tokenizer.encode('[BOS] The King must leave the throne now . [EOS]',
                      return_tensors='pt')
# set top_k = 50 and set top_p = 0.95
final_outputs = model.generate(
    ids,
    do_sample=True,
    max_length=300,
    top_k=40,
    top_p=0.95,
)

print("Output:\n" + 100 * '-')
for i, final_output in enumerate(final_outputs):
    print("{}: {}".format(i, tokenizer.decode(final_output, skip_special_tokens=True)))
