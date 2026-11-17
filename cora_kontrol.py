"""Hafta 2 kontrolü: GCN + TS + ISTS, Cora, 5 bölme x 3 kat.
Hedef (ISTS makalesi, Tablo 2, NLL x100): Uncal 62.9, TS 56.4, ISTS 55.8.
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from bapkanit.veri import yukle
from bapkanit.omurga import egit_gcn
from bapkanit import kalibrasyon as K, olcut

ap = argparse.ArgumentParser()
ap.add_argument("--veri", default="Cora")
ap.add_argument("--kok", default="../veri")
ap.add_argument("--bolmeler", default="../GATS/data/split")
ap.add_argument("--bolme_sayisi", type=int, default=5)
ap.add_argument("--cikti", default="cora_kontrol.json")
a = ap.parse_args()
torch.set_num_threads(2)

sonuc = []
for bol in range(a.bolme_sayisi):
    for kat in range(3):
        t0 = time.time()
        v = yukle(a.veri, bol, kat, a.kok, a.bolmeler)
        m = egit_gcn(v, tohum=bol * 3 + kat)
        with torch.no_grad():
            z = m(v.x, v.edge_index)
        c, t = v.cal_mask, v.test_mask
        phi = K.standartlastir(K.betimleyiciler(z, v.edge_index, v.y, v.train_mask))
        T = K.ts_uydur(z[c], v.y[c])
        ist = K.ists_uydur(z[c], v.y[c], phi[c], tohum=bol * 3 + kat)
        Ti = K.ists_sicaklik(ist["b"], ist["w"], phi)
        r = dict(bolme=bol, kat=kat, lam=ist["lam"], T=T,
                 uncal=olcut.hepsi(F.log_softmax(z[t], 1), v.y[t]),
                 ts=olcut.hepsi(F.log_softmax(z[t] / T, 1), v.y[t]),
                 ists=olcut.hepsi(F.log_softmax(z[t] / Ti[t].view(-1, 1), 1), v.y[t]))
        sonuc.append(r)
        print(f"bolme {bol} kat {kat} | acc {r['uncal']['acc']:.3f} | NLLx100 uncal {100*r['uncal']['nll']:.1f} "
              f"TS {100*r['ts']['nll']:.1f} ISTS {100*r['ists']['nll']:.1f} (lam {ist['lam']}) | {time.time()-t0:.0f}s", flush=True)

json.dump(sonuc, open(a.cikti, "w"), indent=1)
for k in ["uncal", "ts", "ists"]:
    for o in ["acc", "nll", "brier", "ece", "aurc"]:
        x = np.array([r[k][o] for r in sonuc]) * 100
        print(f"{k:5s} {o:5s} {x.mean():6.2f} ± {x.std():.2f}")
