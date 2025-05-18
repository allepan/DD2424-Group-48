import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
from collections import Counter
from itertools import product
from tqdm import tqdm 


csv_path = r"C:\Users\klapp\Downloads\All-seasons.csv"
df = pd.read_csv(csv_path, encoding='utf-8')
allowed_speakers = {"cartman", "kyle", "stan", "randy", "chef"}
lines = []
for _, row in df.iterrows():
    sp = str(row.iloc[2]).strip().lower()
    if sp in allowed_speakers:
        text_line = str(row.iloc[3]).strip()
        if text_line:
            lines.append(f"{sp}: {text_line}")
text = "\n".join(lines)

ref_words = set(re.findall(r"\w+", text.lower()))
ref_ngrams = {}
for n in [2,3]:
    chars = list(text.replace("\n", " "))
    ref_ngrams[n] = set(tuple(chars[i:i+n]) for i in range(len(chars)-n+1))

unique_chars = sorted(set(text))
char_to_ind = {ch:i for i,ch in enumerate(unique_chars)}
ind_to_char = {i:ch for i,ch in enumerate(unique_chars)}
vocab_size = len(unique_chars)
print(f"Vocab size: {vocab_size}")

data = np.array([char_to_ind[ch] for ch in text], dtype=np.int64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print_interval = 5000

def get_batch(data, seq_length, batch_size):
    if len(data) <= seq_length:
        raise ValueError(f"Dataset too short: length={len(data)} <= seq_length={seq_length}")
    max_start = len(data) - seq_length - 1
    if max_start < 1:
        raise ValueError(f"Sequence length {seq_length} too large for data length {len(data)}")
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    xb = np.stack([data[s:s+seq_length] for s in starts])
    yb = np.stack([data[s+1:s+seq_length+1] for s in starts])
    return torch.tensor(xb, device=device), torch.tensor(yb, device=device)


def sample_text(model, speaker, length, method='nucleus', temperature=0.8, top_k=50, top_p=0.9):
    prefix = f"{speaker}: "
    idxs = torch.tensor([[char_to_ind[ch] for ch in prefix]], device=device)
    generated = []
    with torch.no_grad():
        emb = model['emb'](idxs)
        out, hidden = (model['lstm'] if 'lstm' in model else model['rnn'])(emb, None)
        last = idxs[:, -1:]
        for _ in range(length):
            emb_chr = model['emb'](last)
            out, hidden = (model['lstm'] if 'lstm' in model else model['rnn'])(emb_chr, hidden)
            logits = model['fc'](out[:, -1, :]) / temperature
            probs = F.softmax(logits, dim=-1)
            if method == 'top_k':
                vals, idx = torch.topk(logits, top_k, dim=-1)
                probs_k = F.softmax(vals, dim=-1)
                sel = torch.multinomial(probs_k, 1)
                next_idx = idx.gather(-1, sel)
            else:
                sorted_p, sorted_idx = torch.sort(probs, descending=True)
                cum_p = torch.cumsum(sorted_p, dim=-1)
                mask = cum_p <= top_p
                mask[...,0] = True
                filt = sorted_p * mask
                filt = filt / filt.sum(dim=-1, keepdim=True)
                sel = torch.multinomial(filt, 1)
                next_idx = sorted_idx.gather(-1, sel)
            ch = ind_to_char[int(next_idx)]
            generated.append(ch)
            last = next_idx
    return ''.join(generated)


def compute_metrics(texts):
    def distinct_n(txt,n):
        t=list(txt)
        ng=[tuple(t[i:i+n]) for i in range(len(t)-n+1)]
        return len(set(ng))/len(ng) if ng else 0
    def spell_acc(txt):
        ws=re.findall(r"\w+",txt.lower())
        return sum(w in ref_words for w in ws)/len(ws) if ws else 0
    def overlap(txt,n):
        cs=list(txt.replace("\n"," "))
        ng=[tuple(cs[i:i+n]) for i in range(len(cs)-n+1)]
        return sum(x in ref_ngrams[n] for x in ng)/len(ng) if ng else 0
    res={}
    for key, txts in texts.items():
        res[key] = {
            'distinct1': np.mean([distinct_n(t,1) for t in txts]),
            'distinct2': np.mean([distinct_n(t,2) for t in txts]),
            'spell_acc': np.mean([spell_acc(t) for t in txts]),
            'overlap2': np.mean([overlap(t,2) for t in txts]),
            'overlap3': np.mean([overlap(t,3) for t in txts])
        }
    return res


def train_model(model_type='lstm', num_layers=1, hidden_size=256,
                batch_size=32, lr=1e-3, seq_length=200, max_iters=50000,
                print_interval=print_interval):
    torch.manual_seed(42)
    emb = torch.nn.Embedding(vocab_size, hidden_size).to(device)
    if model_type == 'lstm':
        rnn_mod = torch.nn.LSTM(hidden_size, hidden_size, num_layers,
                                batch_first=True, dropout=(0.2 if num_layers > 1 else 0.0)).to(device)
    else:
        rnn_mod = torch.nn.RNN(hidden_size, hidden_size, num_layers,
                               nonlinearity='tanh', batch_first=True,
                               dropout=(0.2 if num_layers > 1 else 0.0)).to(device)
    fc = torch.nn.Linear(hidden_size, vocab_size).to(device)
    params = list(emb.parameters()) + list(rnn_mod.parameters()) + list(fc.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    crit = torch.nn.CrossEntropyLoss()


    if model_type == 'lstm':
        model_dict = {'emb': emb, 'lstm': rnn_mod, 'fc': fc}
    else:
        model_dict = {'emb': emb, 'rnn': rnn_mod, 'fc': fc}

    smooth = None
    best = float('inf')
    best_state = None
    hidden = None
    rnn_mod.train(); emb.train(); fc.train(); rnn_mod.flatten_parameters()

    smooth_losses = []
    for it in range(1, max_iters + 1):
        xb, yb = get_batch(data, seq_length, batch_size)
        opt.zero_grad()
        out, hidden = rnn_mod(emb(xb), hidden)
        if model_type == 'lstm':
            hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            hidden = hidden.detach()
        logits = fc(out)
        loss = crit(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5)
        opt.step()

        lv = loss.item()
        smooth = lv if smooth is None else 0.99 * smooth + 0.01 * lv
        smooth_losses.append(smooth)
        if smooth < best:
            best = smooth
            best_state = {'emb': emb.state_dict(), 'rnn': rnn_mod.state_dict(), 'fc': fc.state_dict()}

        if it == 1 or (print_interval and it % print_interval == 0):
            print(f"Iter {it} — Smooth Loss: {smooth:.4f}")
            total_chars = 200
            per_sp = total_chars // len(allowed_speakers)
            for sp in allowed_speakers:
                sample_nuc = sample_text(model_dict, sp, per_sp, method='nucleus')
                print(f"{sp.title()}: {sample_nuc}")

    emb.load_state_dict(best_state['emb'])
    rnn_mod.load_state_dict(best_state['rnn'])
    fc.load_state_dict(best_state['fc'])

    final_model = model_dict
    return final_model, smooth_losses



model, smooth_losses = train_model(model_type='lstm', num_layers=1,
                                  hidden_size=256, batch_size=32,
                                  lr=1e-3, seq_length=200,
                                  max_iters=50000)


plt.figure(figsize=(8,5))
plt.plot(range(1, len(smooth_losses)+1), smooth_losses)
plt.xlabel('Iteration')
plt.ylabel('Smooth Loss')
plt.title('Smooth Loss over iterations')
plt.show()


print("Sampling from best model (1000 chars total):")
total_chars = 1000
per_sp = total_chars // len(allowed_speakers)
for sp in allowed_speakers:
    sample = sample_text(model, sp, per_sp, method='nucleus')[:per_sp]
    print(f"{sp.title()}: {sample}")


model_types   = ['rnn', 'lstm']
layer_options = [1, 2]
hidden_sizes  = [128, 256]
batch_sizes   = [16, 32]
learning_rates= [1e-3, 1e-4]
max_iters     = 10000 

param_grid = list(product(model_types, layer_options, hidden_sizes, batch_sizes, learning_rates))
exp_results = []


with tqdm(param_grid, desc="Grid Search", unit="config") as pbar:
    for mtype, nl, hs, bs, lr in pbar:
        key = f"{mtype}_layers{nl}_hs{hs}_bs{bs}_lr{lr}"
        model_cfg, _ = train_model(
            model_type=mtype,
            num_layers=nl,
            hidden_size=hs,
            batch_size=bs,
            lr=lr,
            seq_length=200,
            max_iters= 10000,
            print_interval= 5000 
        )

        txts = {
            'top_k_'   + key: [sample_text(model_cfg, 'stan', 200, method='top_k')   for _ in range(5)],
            'nucleus_' + key: [sample_text(model_cfg, 'stan', 200, method='nucleus') for _ in range(5)]
        }
        mets = compute_metrics(txts)
        for mkey, mvals in mets.items():
            exp_results.append({'config': key, 'method': mkey, **mvals})



import pandas as pd

df_res = pd.DataFrame(exp_results)

print("All Results")
print(df_res)

df_avg = df_res.groupby('config').mean(numeric_only=True)

best_config = df_avg['distinct1'].idxmax()
best_metrics = df_avg.loc[best_config]

print(f"=== Bästa konfiguration enligt Avg Distinct-1: {best_config} ===")
print(f"distinct1    : {best_metrics['distinct1']:.4f}  # Unika tecken-gram (n=1)")
print(f"distinct2    : {best_metrics['distinct2']:.4f}  # Unika tecken-gram (n=2)")
print(f"spell_acc    : {best_metrics['spell_acc']:.4f}  # Stavningsprecision")
print(f"overlap2     : {best_metrics['overlap2']:.4f}  # Andel 2-gram som finns i referens")
print(f"overlap3     : {best_metrics['overlap3']:.4f}  # Andel 3-gram som finns i referens")

plt.figure(figsize=(12,6))
sorted_dist1 = df_avg['distinct1'].sort_values()
colors = ['blue'] * len(sorted_dist1)

pos = list(sorted_dist1.index).index(best_config)
colors[pos] = 'red'
ax = sorted_dist1.plot(kind='bar', color=colors)
plt.xlabel('Config')
plt.ylabel('Avg Distinct-1')
plt.title('Avg Distinct-1 Across Configurations (bästa markerad i rött)')
plt.xticks(rotation=45, ha='right')

best_val = sorted_dist1.iloc[pos]
ax.annotate('Bästa här', xy=(pos, best_val),
            xytext=(pos, best_val + 0.005),
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            ha='center')

label_text = f"d1={best_metrics['distinct1']:.3f}"
label_text += f"d2={best_metrics['distinct2']:.3f}"
label_text += f"spell={best_metrics['spell_acc']:.3f}"
ax.text(pos, best_val + 0.01, label_text, ha='center')
plt.tight_layout()
plt.show()

