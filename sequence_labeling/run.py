import os

os.system("python main.py --phase cmed.dl.base_model.norm")
os.system("python main.py --phase cmed.dl.bilstm.norm")
os.system("python main.py --phase cmed.dl.cnn.norm")
os.system("python main.py --phase cmed.dl.gru.norm")
os.system("python main.py --phase cmed.dl.mlp.norm")
os.system("python main.py --phase cmed.dl.rnn.norm")
os.system("python main.py --phase cmed.dl.s2s.norm")
os.system("python main.py --phase cmed.dl.s2s_dotproduct_attn.norm")
os.system("python main.py --phase cmed.dl.self_attn.norm")
os.system("python main.py --phase cmed.dl.self_attn_multihead.norm")
