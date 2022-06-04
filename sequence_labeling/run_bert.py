import os

os.system("python main.py --phase cmed.dl.bert_lstm.norm")
os.system("python main.py --phase cmed.dl.bert_bilstm.norm")
os.system("python main.py --phase cmed.dl.bert_cnn.norm")
os.system("python main.py --phase cmed.dl.bert_gru.norm")
os.system("python main.py --phase cmed.dl.bert_mlp.norm")
os.system("python main.py --phase cmed.dl.bert_rnn.norm")
os.system("python main.py --phase cmed.dl.bert_s2s.norm")
os.system("python main.py --phase cmed.dl.bert_s2s_attn.norm")
os.system("python main.py --phase cmed.dl.bert_self_attn.norm")
os.system("python main.py --phase cmed.dl.bert_self_attn_multihead.norm")
