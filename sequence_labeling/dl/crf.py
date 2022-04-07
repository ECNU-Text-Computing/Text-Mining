#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
CRF
======
A class for CRF.
参考：
《线性链条件随机场(CRF)的原理与实现》，https://www.cnblogs.com/weilonghu/p/11960984.html
《LSTM+CRF 解析（代码篇）》，https://zhuanlan.zhihu.com/p/97858739（实现batch，mask）
"""
import torch
import torch.nn as nn


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
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class CRF(nn.Module):
    def __init__(self, tagset_size, tag_to_ix):
        super(CRF, self).__init__()
        # tag_to_ix已添加"<START>"、"<STOP>"
        self.tag_to_ix = tag_to_ix
        self.tagset_size = tagset_size

        # 转移矩阵，矩阵中代表由状态j到i的概率
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 规则：任何状态都不能转移到'<START>','<STOP>'不能转移到其他状态
        self.transitions.data[self.tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, self.tag_to_ix["<STOP>"]] = -10000

    # 计算配分函数Z(x)
    # Z(x)的作用：做全局归一化，解决标注偏置问题。其关键在于需要遍历所有路径。
    # Z(x)的计算：前向算法
    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix["<START>"]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # 初始状态的forward_var
        forward_var = init_alphas

        # 迭代整个句子
        for feat in feats:  # feats为输入序列，即x
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):  # 6
                # broadcast the emission score: it is the same regardless of
                # the previous tag 1*6 emit_score的6个值相同
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i 1*6 从i到下一个tag的概率
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]  # 到第（t-1）step时6个标签的各自分数
        alpha = log_sum_exp(terminal_var)
        return alpha

    # 计算给定输入序列和标签序列的匹配函数，即s(x,y)
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix["<START>"]], dtype=torch.long), tags])  # 将START_TAG的标签拼接到tag序列上
        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]]：第i时间步对应的标签转移到下一时间步对应标签的概率
            # feat[tags[i+1]]：feats第i个时间步对应标签的score。之所以用i+1是要跳过tag序列开头的start
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix["<STOP>"], tags[-1]]
        return score

    # 维特比解码，给定输入x和相关参数(发射矩阵和转移矩阵)，获得概率最大的标签序列
    def _viterbi_decode(self, feats):  # 维特比
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
                next_tag_var = forward_var + self.transitions[next_tag]  # 上一时刻的forward_var与transition矩阵相作用
                best_tag_id = argmax(next_tag_var)  # 选取上一时刻的tag使得到当前时刻的某个tag的路径分数最大
                bptrs_t.append(best_tag_id)  # 添加路径,注意此时的best_tag_id指向的是上一时刻的label
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

    # 损失函数 = Z(x) - s(x,y)
    def neg_log_likelihood(self, feats, tags):
        forward_score = self._forward_alg(feats)  # 计算配分函数Z(x)
        gold_score = self._score_sentence(feats, tags)  # 根据真实的tags计算score
        return forward_score - gold_score
