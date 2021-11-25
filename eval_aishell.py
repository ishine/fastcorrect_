import sys
import torch
import argparse
import re
from fairseq.models.transformer import TransformerModel
import os
import os.path
import time
import json
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def remove_ch_spaces(input_str):
    return re.sub(r"(?<=[\u4e00-\u9fff])(\s+)(?=[\u4e00-\u9fff])", "", input_str.strip())


try:
    model_name_or_path = sys.argv[3]
except:
    model_name_or_path = "checkpoints/shared_baseline"

try:
    iter_decode_max_iter = int(sys.argv[4])
except:
    iter_decode_max_iter = -1

checkpoint_file = "checkpoint_best.pt"

print("test {}/{}".format(model_name_or_path, checkpoint_file))
data_name_or_path = "./data"
bpe = "sentencepiece"
sentencepiece_model = "./sentence.bpe.model"

commonset_dir = "./eval_data"
res_dir = os.path.join(model_name_or_path, ("results_aishell" if (iter_decode_max_iter == -1) else ("results_aishell_b" + str(iter_decode_max_iter))))
tmp_dir = os.path.join(model_name_or_path, ("tmp_aishell" if (iter_decode_max_iter == -1) else ("tmp_aishell_b" + str(iter_decode_max_iter))))
os.makedirs(res_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

try:
    short_set = sys.argv[1].split(',')
except:
    raise ValueError()

print("short_set:", short_set)

transf_gec = TransformerModel.from_pretrained(model_name_or_path, checkpoint_file=checkpoint_file, data_name_or_path=data_name_or_path, bpe=bpe, sentencepiece_model=sentencepiece_model)
transf_gec.eval()
transf_gec.cuda()

for input_tsv in [os.path.join(commonset_dir, f, "data.json") for f in short_set]:
    all_time = []
    eval_origin_dict = json.load(open(input_tsv, 'r', encoding='utf-8'))
    translate_input_dict = {}
    for k, v in eval_origin_dict["utts"].items():
        translate_input_dict[k] = (v["output"][0]["rec_text"].replace('<eos>', '').strip(), v["output"][0]["token"])
    translated_output_dict = {}
    for k, v in translate_input_dict.items():
        text = v[0]
        gt = v[1]
        start_time = time.time()
        time_ok = False
        try:
            if iter_decode_max_iter != -1:
                translated = transf_gec.translate(text, iter_decode_max_iter=iter_decode_max_iter)
            else:
                translated = transf_gec.translate(text)
            if isinstance(translated, tuple):
                all_time.append(translated[1])
                time_ok = True
                translated = translated[0]
            translated_output_dict[k] = (text, gt, translated)
        except Exception as e:
            print(input_tsv + "\t" + text + "\n")
            translated_list.append(text)
            raise e
            continue 
        end_time = time.time()
        if not time_ok:
            all_time.append(end_time - start_time)
        eval_origin_dict["utts"][k]["output"][0]["rec_text"] = " ".join("".join(translated.split()).replace('▁', ' ').strip().split())
        translated_char = [i for i in eval_origin_dict["utts"][k]["output"][0]["rec_text"]]
        eval_origin_dict["utts"][k]["output"][0]["rec_token"] = " ".join(translated_char)
    os.makedirs(os.path.join(res_dir, input_tsv.split('/')[-2]), exist_ok=True)
    with open(os.path.join(res_dir, input_tsv.split('/')[-2], input_tsv.split('/')[-2] + "_time.txt"), 'w') as outfile:
        outfile.write("{}\t{}\t{}\n".format(len(all_time), sum(all_time), sum(all_time)/len(all_time)))
    json.dump(eval_origin_dict, open(os.path.join(res_dir, input_tsv.split('/')[-2], 'data.json'), 'w', encoding='utf-8'), indent=4, sort_keys=True, ensure_ascii=False)
