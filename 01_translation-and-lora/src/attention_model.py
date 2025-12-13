import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, enc_size, dec_size, hid_size, activ=torch.tanh):
        super().__init__()
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.hid_size = hid_size
        self.activ = activ

        self.linear_encoder = nn.Parameter(
            torch.rand(self.enc_size, self.hid_size), requires_grad=True
        )
        self.linear_decoder = nn.Parameter(
            torch.rand(self.dec_size, self.hid_size), requires_grad=True
        )
        self.linear_out = nn.Parameter(torch.rand(self.hid_size, 1), requires_grad=True)

    def forward(self, enc, dec, inp_mask):
        encoded = enc @ self.linear_encoder
        decoded = dec @ self.linear_decoder
        result = self.activ(encoded + decoded.unsqueeze(1))
        logits = (result @ self.linear_out).squeeze(2)
        logits[inp_mask == 0] = -1e9
        probs = F.softmax(logits, dim=1)
        attn = (enc * probs.unsqueeze(-1)).sum(dim=1)
        return attn, probs


class AttentiveModel(nn.Module):
    def __init__(self, inp_voc, out_voc, emb_size=64, hid_size=128, attn_size=128):
        super().__init__()
        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)
        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True)
        self.dec_start = nn.Linear(hid_size, hid_size)
        self.attention_layer = AttentionLayer(hid_size, hid_size, attn_size)
        self.dec0 = nn.GRUCell(emb_size + hid_size, hid_size)
        self.logits = nn.Linear(hid_size, len(out_voc))

    def encode(self, inp):
        inp_emb = self.emb_inp(inp)
        enc_seq, _ = self.enc0(inp_emb)

        mask = inp != self.inp_voc.eos_ix
        lengths = mask.to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]

        dec_start = self.dec_start(last_state)
        first_attn_probas = self.attention_layer(enc_seq, dec_start, mask)[1]

        return [dec_start, enc_seq, mask, first_attn_probas]

    def decode_step(self, prev_state, prev_tokens):
        prev_gru0, encoder_sequence, mask, _ = prev_state
        attn_response, attn_probas = self.attention_layer(
            encoder_sequence, prev_gru0, mask
        )
        out_emb = self.emb_out(prev_tokens)
        new_dec_state = self.dec0(torch.cat([out_emb, attn_response], dim=1), prev_gru0)
        output_logits = self.logits(new_dec_state)
        return [new_dec_state, encoder_sequence, mask, attn_probas], output_logits
