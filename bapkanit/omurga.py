"""GATS ile aynı GCN omurgası ve eğitim düzeni (lr 0.01, wd 5e-4, dropout 0.5, sabır 100)."""
import copy
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, giris, sinif, gizli=64, dropout=0.5):
        super().__init__()
        self.c1, self.c2, self.p = GCNConv(giris, gizli), GCNConv(gizli, sinif), dropout

    def forward(self, x, ei, edge_weight=None):
        x = F.dropout(F.relu(self.c1(x, ei, edge_weight)), self.p, self.training)
        return self.c2(x, ei, edge_weight)


def egit_gcn(veri, tohum, donem=2000, sabir=100):
    torch.manual_seed(tohum)
    m = GCN(veri.num_features, veri.num_classes)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    en_iyi, vacc_mx, vlss_mn, adim = None, 0.0, float("inf"), 0
    for _ in range(donem):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(veri.x, veri.edge_index)[veri.train_mask], veri.y[veri.train_mask]).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            lp = F.log_softmax(m(veri.x, veri.edge_index), 1)
        vm = veri.cal_mask
        vacc = (lp[vm].argmax(1) == veri.y[vm]).float().mean().item()
        vlss = -lp[vm, veri.y[vm]].mean().item()
        if vacc >= vacc_mx or vlss <= vlss_mn:
            if vacc >= vacc_mx and vlss <= vlss_mn:
                en_iyi = copy.deepcopy(m.state_dict())
            vacc_mx, vlss_mn, adim = max(vacc, vacc_mx), min(vlss, vlss_mn), 0
        else:
            adim += 1
            if adim >= sabir:
                break
    m.load_state_dict(en_iyi)
    m.eval()
    return m
