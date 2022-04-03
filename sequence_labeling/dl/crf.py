#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
CRF
======
A class for something.
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def argmax(vec):
    # return the argmax as a python int
    # 返回行tensor最大值
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
# 计算离差平方和
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class CRF(nn.Module):
    def __init__(self, tagset_size, tag_to_ix):
        super(CRF, self).__init__()
        self.tag_to_ix = {"B": 0, "I": 1, "O": 2, "<START>": 3, "<STOP>": 4}  # 词典转化
        self.tagset_size = len(self.tag_to_ix)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # 转移矩阵，矩阵中代表由j到i的概率，5*5
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 减小转移矩阵中的概率
        self.transitions.data[self.tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, self.tag_to_ix["<STOP>"]] = -10000

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix["<START>"]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # 初始状态的forward_var
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:  # tag输出层结果，5维
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):  # 5
                # broadcast the emission score: it is the same regardless of
                # the previous tag 1*5 emit_score的5个值相同
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i 1*5 从i到下一个tag的概率
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]  # 到第（t-1）step时５个标签的各自分数
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix["<START>"]], dtype=torch.long), tags])  # 将START_TAG的标签３拼接到tag序列上
        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix["<STOP>"], tags[-1]]
        return score

    def _viterbi_decode(self, feats):  #维特比
        backpointers = []

        # Initialize the viterbi variables in log space 初始化
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix["<START>"]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]  # 其他标签（B,I,O,Start,End）到标签next_tag的概率
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)  # 从step0到step(i-1)时5个序列中每个序列的最大score
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):  # 从后向前走，找到一个best路径
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix["<START>"]  # Sanity check
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path
    # 前向传播
    def neg_log_likelihood(self, feats, tags):

        forward_score = self._forward_alg(feats)  # 前向传播计算score
        gold_score = self._score_sentence(feats, tags)  # 根据真实的tags计算score
        return forward_score - gold_score