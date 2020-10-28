# import torch
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt2large")
# model = AutoModel.from_pretrained("sberbank-ai/rugpt2large")

# # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
# input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  
# with torch.no_grad():
#     last_hidden_states = model(input_ids)[0]
#     print(last_hidden_states)

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt2large")
#AutoModelForCausalLM AutoModelForSeq2SeqLM AutoModelForMaskedLM
model = AutoModelWithLMHead.from_pretrained("sberbank-ai/rugpt2large")
model.eval()
model.to('cuda')

def generate(text):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.to('cuda')
    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(greedy_output[0], skip_special_tokens=True)

'''Пётр I Алексе́евич, прозванный Вели́ким (30 мая [9 июня] 1672 года — 28 января [8 февраля] 1725 года) — последний царь всея Руси (с 1682 года) и первый Император Всероссийский (с 1721 года). Представитель династии Романовых. Был провозглашён царём в 10-летнем возрасте, стал править самостоятельно с 1689 года. Формальным соправителем Петра был его брат Иван (до своей смерти в 1696 году). С юных лет проявляя интерес к наукам и заграничному образу жизни,'''
s = generate('Маньяк читает книгу про глубокое обучение')
print(s)