import os
import json

config_list = ["cmed.dl.bert_lstm.norm", "cmed.dl.bert_bilstm.norm",
               "cmed.dl.bert_cnn.norm", "cmed.dl.bert_gru.norm",
               "cmed.dl.bert_mlp.norm", "cmed.dl.bert_rnn.norm",
               "cmed.dl.bert_s2s.norm", "cmed.dl.bert_s2s_attn.norm",
               "cmed.dl.bert_self_attn.norm", "cmed.dl.bert_self_attn_multihead.norm"]

for item in config_list:
    config_file_path = './config/cmed/dl/{}.json'.format(item)
    with open(config_file_path, 'r', encoding='utf-8') as cf:
        config_dict = json.load(cf)
        config_dict["checkpoint"] = 'hfl/chinese-bert-wwm-ext'

    with open(config_file_path, 'w', encoding='utf-8') as cf:
        json.dump(config_dict, cf)

    cmd_str = "python main.py --phase {}".format(item)

    os.system(cmd_str)
