import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, enc_size, dec_size, hid_size, activ=torch.tanh):
        super().__init__()
        self.activ = activ

        self.linear_encoder = nn.Linear(enc_size, hid_size, bias=False)
        self.linear_decoder = nn.Linear(dec_size, hid_size, bias=False)
        self.linear_out = nn.Linear(hid_size, 1, bias=False)

    def forward(self, enc, dec, inp_mask):
        encoded = self.linear_encoder(enc)
        decoded = self.linear_decoder(dec)
        result = self.activ(encoded + decoded.unsqueeze(1))
        logits = self.linear_out(result).squeeze(2)
        
        logits.masked_fill_(~inp_mask, -1e9)
        probs = F.softmax(logits, dim=1)
        attn = (enc * probs.unsqueeze(-1)).sum(dim=1)
        return attn, probs


class AttentiveModel(nn.Module):
    def __init__(self, inp_voc, out_voc, emb_size=64, hid_size=128, attn_size=128):
        super().__init__()
        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)
        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True, bidirectional=True)
        self.dec_start = nn.Linear(hid_size * 2, hid_size)
        self.attention_layer = AttentionLayer(hid_size * 2, hid_size, attn_size)
        self.dec0 = nn.GRUCell(emb_size + hid_size * 2, hid_size)
        self.logits = nn.Linear(hid_size, len(out_voc))

    def forward(self, inp, out):
        state = self.encode(inp)
        logits_sequence = []

        batch_size = out.shape[0]
        current_tokens = torch.full((batch_size,), self.out_voc.bos_ix, dtype=torch.long, device=out.device)

        for i in range(out.shape[1]):
            state, logits = self.decode_step(state, current_tokens)
            logits_sequence.append(logits)
            current_tokens = out[:, i]
            
        return torch.stack(logits_sequence, dim=1)

    def decode(self, initial_state, out_voc, max_len=100):
        batch_size = initial_state[0].shape[0]
        state = initial_state
        probs_history = [state[-1]]
        
        current_tokens = torch.full((batch_size,), out_voc.bos_ix, dtype=torch.long, device=initial_state[0].device)
        decoded_tokens = []

        for i in range(max_len):
            state, logits = self.decode_step(state, current_tokens)
            probs_history.append(state[-1])
            
            current_tokens = logits.argmax(dim=-1)
            decoded_tokens.append(current_tokens)
            
        return torch.stack(decoded_tokens, dim=1), torch.stack(probs_history, dim=1)
    
    def encode(self, inp):
        inp_emb = self.emb_inp(inp)
        enc_seq, _ = self.enc0(inp_emb)

        mask = inp != self.inp_voc.eos_ix
        lengths = mask.to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        
        batch_idx = torch.arange(inp.shape[0], device=inp.device)
        hid_size = self.dec_start.in_features // 2
        
        last_state_fwd = enc_seq[batch_idx, lengths, :hid_size]
        last_state_bwd = enc_seq[batch_idx, 0, hid_size:]
        
        last_state = torch.cat([last_state_fwd, last_state_bwd], dim=-1)
        dec_start = self.dec_start(last_state)

        first_eos_mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=1) & ~mask
        attn_mask = mask | first_eos_mask

        first_attn_probas = self.attention_layer(enc_seq, dec_start, attn_mask)[1]

        return[dec_start, enc_seq, attn_mask, first_attn_probas]

    def decode_step(self, prev_state, prev_tokens):
        prev_gru0, encoder_sequence, mask, _ = prev_state
        attn_response, attn_probas = self.attention_layer(
            encoder_sequence, prev_gru0, mask
        )
        out_emb = self.emb_out(prev_tokens)
        new_dec_state = self.dec0(torch.cat([out_emb, attn_response], dim=1), prev_gru0)
        output_logits = self.logits(new_dec_state)
        return [new_dec_state, encoder_sequence, mask, attn_probas], output_logits

    def translate_lines(self, inp_lines, max_len=100):
        device = next(self.parameters()).device
        inp = self.inp_voc.to_matrix(inp_lines).to(device)
        initial_state = self.encode(inp)
        decoded_tokens, probs_history = self.decode(initial_state, self.out_voc, max_len)
        return self.out_voc.to_lines(decoded_tokens.cpu().numpy()), probs_history
