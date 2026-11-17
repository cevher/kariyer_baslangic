"""Hafta 2 kontrolü: düğüm düzeyi GSAT.
(1) BA-Shapes: açıklama ROC AUC (bilinen motif kenarları) + doğruluk.
(2) Cora (GATS bölmeleri): doğruluk düz GCN'e (~82.8) yakın mı.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from bapkanit.veri import yukle, yukle_sentetik
from bapkanit.gsat_dugum import egit_gsat
from bapkanit import olcut

ap = argparse.ArgumentParser()
ap.add_argument("--sentetik", default="../DIG_storage/xgraph/datasets/BA_shapes.pkl")
ap.add_argument("--kok", default="../veri")
ap.add_argument("--bolmeler", default="../GATS/data/split")
ap.add_argument("--tohum_sayisi", type=int, default=3)
ap.add_argument("--katman_sentetik", type=int, default=3)
a = ap.parse_args()
torch.set_num_threads(2)
sonuc = {"ba_shapes": [], "cora": []}

v = yukle_sentetik(a.sentetik)
for t in range(a.tohum_sayisi):
    m, g = egit_gsat(v, tohum=t, katman=a.katman_sentetik, dropout=0.0, wd=0.0, donem=2000, sabir=300)
    with torch.no_grad():
        z, p, att = m(v.x, g, ornekleme=False)
    # g["ei"] ile v.edge_index aynı kenar kümesi; etiketi g sırasına taşı
    n = v.num_nodes
    an_v = v.edge_index[0] * n + v.edge_index[1]
    an_g = g["ei"][0] * n + g["ei"][1]
    s = torch.argsort(an_v)
    et = v.kenar_etiket[s][torch.searchsorted(an_v[s], an_g)]
    motif_dugum = v.y > 0
    yerel = motif_dugum[g["ei"][0]] | motif_dugum[g["ei"][1]]
    r = dict(tohum=t, test_acc=(z[v.test_mask].argmax(1) == v.y[v.test_mask]).float().mean().item(),
             auc_tum=roc_auc_score(et.numpy(), att.numpy()),
             auc_motif_cevresi=roc_auc_score(et[yerel].numpy(), att[yerel].numpy()))
    sonuc["ba_shapes"].append(r); print("BA-Shapes", r, flush=True)

for bol in range(a.tohum_sayisi):
    v = yukle("Cora", bol, 0, a.kok, a.bolmeler)
    m, g = egit_gsat(v, tohum=bol)
    with torch.no_grad():
        z, p, att = m(v.x, g, ornekleme=False)
    r = dict(bolme=bol, **olcut.hepsi(F.log_softmax(z[v.test_mask], 1), v.y[v.test_mask]),
             p_ort=p.mean().item(), p_gt05=(p > 0.5).float().mean().item())
    sonuc["cora"].append(r); print("Cora", r, flush=True)
json.dump(sonuc, open("gsat_kontrol.json", "w"), indent=1)
