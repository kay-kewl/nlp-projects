import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu


class Vocab:
    def __init__(self, tokens, bos_token="_BOS_", eos_token="_EOS_", unk_token="_UNK_"):
        self.tokens = [bos_token, eos_token, unk_token] + sorted(tokens)
        self.token_to_ix = {t: i for i, t in enumerate(self.tokens)}
        self.bos_ix = self.token_to_ix[bos_token]
        self.eos_ix = self.token_to_ix[eos_token]
        self.unk_ix = self.token_to_ix[unk_token]

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def from_lines(lines, tokenizer=str.split):
        tokens = set()
        for line in lines:
            tokens.update(tokenizer(line))
        return Vocab(list(tokens))

    def tokenize(self, string):
        if not isinstance(string, str):
            return list(string)
        return string.split()

    def to_matrix(self, lines, max_len=None, dtype="int32", batch_first=True):
        lines = [self.tokenize(line) for line in lines]
        max_len = max_len or max(map(len, lines)) + 1
        matrix = np.full((len(lines), max_len), self.eos_ix, dtype=dtype)
        for i, seq in enumerate(lines):
            row_len = min(len(seq), max_len - 1)
            matrix[i, :row_len] = [
                self.token_to_ix.get(w, self.unk_ix) for w in seq[:row_len]
            ]
        return torch.as_tensor(matrix) if batch_first else torch.as_tensor(matrix.T)

    def to_lines(self, matrix):
        lines = []
        for row in matrix:
            line = []
            for ix in row:
                if ix == self.eos_ix:
                    break
                line.append(self.tokens[ix])
            lines.append(" ".join(line))
        return lines

    def compute_mask(self, matrix):
        return matrix != self.eos_ix


def compute_loss(model, inp, out, **flags):
    logits_seq = model(inp, out)
    
    mask = out != model.out_voc.eos_ix
    first_eos_mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=1) & ~mask
    mask = mask | first_eos_mask
    
    loss = F.cross_entropy(
        logits_seq.reshape(-1, len(model.out_voc)), 
        out.long().reshape(-1), 
        reduction='none'
    )
    
    return (loss * mask.reshape(-1)).sum() / mask.sum()


def compute_bleu(model, inp_lines, out_lines, bpe_sep="@@ ", **flags):
    device = next(model.parameters()).device
    with torch.no_grad():
        translations, _ = model.translate_lines(inp_lines, max_len=100)
        translations = [line.replace(bpe_sep, "") for line in translations]
        
        actual =[]
        for line in out_lines:
            if not isinstance(line, str):
                line = " ".join(line)
            actual.append(line.replace(bpe_sep, ""))

        return (
            corpus_bleu(
                [[ref.split()] for ref in actual],
                [trans.split() for trans in translations],
                smoothing_function=lambda precisions, **kw:[
                    p + 1.0 / p.denominator for p in precisions
                ],
            )
            * 100
        )
