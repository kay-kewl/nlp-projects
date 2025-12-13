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
    """
    Compute loss (float32 scalar)
    """
    mask = model.out_voc.compute_mask(out)
    targets_one_hot = F.one_hot(out, len(model.out_voc)).to(torch.float32)

    logits_seq = model(inp, out)
    logprobs_seq = torch.log_softmax(logits_seq, dim=-1)
    logp_out = (logprobs_seq * targets_one_hot).sum(dim=-1)

    return -(logp_out * mask).sum() / mask.sum()


def compute_bleu(model, inp_lines, out_lines, bpe_sep="@@ ", **flags):
    device = next(model.parameters()).device
    with torch.no_grad():
        translations, _ = model.translate_lines(inp_lines, max_len=100)
        translations = [line.replace(bpe_sep, "") for line in translations]
        actual = [line.replace(bpe_sep, "") for line in out_lines]
        return (
            corpus_bleu(
                [[ref.split()] for ref in actual],
                [trans.split() for trans in translations],
                smoothing_function=lambda precisions, **kw: [
                    p + 1.0 / p.denominator for p in precisions
                ],
            )
            * 100
        )
