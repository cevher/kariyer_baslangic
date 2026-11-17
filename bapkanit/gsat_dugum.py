"""Düğüm düzeyi GSAT (Miao ve ark., ICML 2022'nin düğüm sınıflamaya uyarlaması).

- Tek GCN hem gömme (tam graf) hem tahmin (maskeli graf) için kullanılır (resmi koddaki gibi).
- Kenar skoru: MLP([h_u, h_v]) -> logit; eğitimde concrete/Gumbel örnekleme, değerlendirmede sigmoid.
- Yönsüz grafta iki yönün maskesi ortalanır.
- Maske mesajı çarpar: GCN normalizasyonu ORİJİNAL graftan bir kez hesaplanır, kenar ağırlığı
  w_uv * a_uv olur, öz-döngüler maskelenmez. (GCN'in maskeli grafa göre yeniden normalize etmesi
  maskenin etkisini kısmen sileceği için bilinçli olarak böyle yapıldı.)
- Kayıp: CE(eğitim düğümleri) + beta * ortalama KL(Bern(p_uv) || Bern(r)), r 0.9'dan 10 dönemde 0.1 iner.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_undirected, coalesce


def hazirla_graf(ei, n):
    """Yönsüz, tekrarsız kenarlar + öz-döngüler; GCN normalizasyonu ve ters-kenar eşlemesi."""
    ei = coalesce(to_undirected(ei, num_nodes=n), num_nodes=n)
    ei = ei[:, ei[0] != ei[1]]
    m = ei.shape[1]
    ei_tam, w = gcn_norm(ei, None, n, add_self_loops=True)
    oz = ei_tam[0] == ei_tam[1]
    # ei_tam içinde öz-döngü olmayan kenarları ei sırasına hizala
    anahtar = ei[0] * n + ei[1]
    anahtar_tam = ei_tam[0] * n + ei_tam[1]
    sira = torch.argsort(anahtar_tam)
    konum = sira[torch.searchsorted(anahtar_tam[sira], anahtar)]
    ters = torch.argsort(anahtar)[torch.searchsorted(anahtar[torch.argsort(anahtar)], ei[1] * n + ei[0])]
    return dict(ei=ei, ei_tam=ei_tam, w=w, oz=oz, konum=konum, ters=ters, m=m, n=n)


class GCNOmurga(nn.Module):
    def __init__(self, giris, sinif, gizli=64, katman=2, dropout=0.5):
        super().__init__()
        boy = [giris] + [gizli] * (katman - 1) + [sinif]
        self.convs = nn.ModuleList([GCNConv(a, b, normalize=False) for a, b in zip(boy[:-1], boy[1:])])
        self.p = dropout

    def forward(self, x, ei, w, gomme=False):
        for i, c in enumerate(self.convs):
            if gomme and i == len(self.convs) - 1:
                return x
            x = c(x, ei, w)
            if i < len(self.convs) - 1:
                x = F.dropout(F.relu(x), self.p, self.training)
        return x


class GSATDugum(nn.Module):
    def __init__(self, giris, sinif, gizli=64, katman=2, dropout=0.5, cikarici_dropout=0.5):
        super().__init__()
        self.clf = GCNOmurga(giris, sinif, gizli, katman, dropout)
        self.cik = nn.Sequential(nn.Linear(2 * gizli, gizli), nn.ReLU(), nn.Dropout(cikarici_dropout),
                                 nn.Linear(gizli, 1))

    def kenar_logit(self, x, g):
        h = self.clf(x, g["ei_tam"], g["w"], gomme=True)
        s, d = g["ei"]
        return self.cik(torch.cat([h[s], h[d]], 1)).squeeze(1)

    @staticmethod
    def ornekle(logit, egitim, sicaklik=1.0):
        if not egitim:
            return logit.sigmoid()
        u = torch.empty_like(logit).uniform_(1e-10, 1 - 1e-10)
        return ((logit + u.log() - (1 - u).log()) / sicaklik).sigmoid()

    def maske_ile(self, x, g, a):
        a = (a + a[g["ters"]]) / 2
        w = g["w"].clone()
        w[g["konum"]] = w[g["konum"]] * a
        return self.clf(x, g["ei_tam"], w), a

    def forward(self, x, g, ornekleme):
        logit = self.kenar_logit(x, g)
        a = self.ornekle(logit, ornekleme)
        z, a = self.maske_ile(x, g, a)
        return z, logit.sigmoid(), a


def r_takvimi(donem, bas=0.9, adim=0.1, aralik=10, son=0.7):
    return max(bas - (donem // aralik) * adim, son)


def bilgi_kaybi(p, r):
    p = p.clamp(1e-6, 1 - 1e-6)
    return (p * (p / r).log() + (1 - p) * ((1 - p) / (1 - r)).log()).mean()


def egit_gsat(veri, tohum, katman=2, beta=1.0, son_r=0.7, donem=1000, sabir=100, lr=0.01, wd=5e-4, dropout=0.5):
    """Erken durdurma GATS kuralıyla, kalibrasyon katında; r takvimi bitmeden durdurulmaz."""
    torch.manual_seed(tohum)
    g = hazirla_graf(veri.edge_index, veri.num_nodes)
    m = GSATDugum(veri.num_features, veri.num_classes, katman=katman, dropout=dropout)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    r_bitis = int(round((0.9 - son_r) / 0.1)) * 10
    en_iyi, vacc_mx, vlss_mn, adim = None, 0.0, float("inf"), 0
    for e in range(donem):
        r = r_takvimi(e, son=son_r)
        m.train(); opt.zero_grad()
        z, p, _ = m(veri.x, g, ornekleme=True)
        kayip = F.cross_entropy(z[veri.train_mask], veri.y[veri.train_mask]) + beta * bilgi_kaybi(p, r)
        kayip.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lp = F.log_softmax(m(veri.x, g, ornekleme=False)[0], 1)
        vm = veri.cal_mask
        vacc = (lp[vm].argmax(1) == veri.y[vm]).float().mean().item()
        vlss = -lp[vm, veri.y[vm]].mean().item()
        if e < r_bitis:
            continue
        if vacc >= vacc_mx or vlss <= vlss_mn:
            if vacc >= vacc_mx and vlss <= vlss_mn:
                en_iyi = copy.deepcopy(m.state_dict())
            vacc_mx, vlss_mn, adim = max(vacc, vacc_mx), min(vlss, vlss_mn), 0
        else:
            adim += 1
            if adim >= sabir:
                break
    m.load_state_dict(en_iyi); m.eval()
    return m, g
