import sophon.sail as sail
from transformers import AutoTokenizer
import numpy as np
import torch

bmodel_path = './models/bm1684x/bce-reranker-base_v1.bmodel'
dev_id = 0

engine = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
graph_name = engine.get_graph_names()[0]
input_names = engine.get_input_names(graph_name)
output_names = engine.get_output_names(graph_name)

sentence_pairs = [['apples', 'I like apples'],['apples', 'I like oranges'],['apples', 'Apples and oranges are fruits']]
print('input sentence_pairs: ', sentence_pairs)
tokenizer = AutoTokenizer.from_pretrained('./token_config')

inputs = tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")

input_ids = inputs['input_ids']
print('input tokens: ', input_ids)
attention_mask = inputs['attention_mask']
input_ids, attention_mask = input_ids.numpy(), attention_mask.numpy()

if input_ids.shape[1] > 512:
    input_ids = input_ids[:, :512]
    attention_mask = attention_mask[:, :512]
elif input_ids.shape[1] < 512:
    input_ids = np.pad(input_ids,
                        ((0, 0), (0, 512 - input_ids.shape[1])),
                        mode='constant', constant_values=0)
    attention_mask = np.pad(attention_mask,
                            ((0, 0), (0, 512 - attention_mask.shape[1])),
                            mode='constant', constant_values=0)
    
input_data = { input_names[0]: input_ids, input_names[1]: attention_mask }
outputs = engine.process(graph_name, input_data)

scores = torch.sigmoid(torch.from_numpy(outputs[output_names[0]].flatten()))

print('reranker scores', scores)


