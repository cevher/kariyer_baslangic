"""Veri yükleme: PyG veri kümeleri + GATS'ın açık bölmeleri (5 bölme x 3 kat, 5_3f_85).

Kat k kalibrasyon kümesi (GATS'ta "val"), kalan iki kat eğitim, test_indices test kümesidir.
"""
from pathlib import Path
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import index_to_mask

GELISTIRME = ["Cora", "Pubmed", "Computers"]
DOGRULAMA = ["Photo", "CS", "CoraFull"]  # kod dondurulmadan çalıştırılmaz (ön-kayıt)


def _yukle_ham(ad, kok):
    t = NormalizeFeatures()
    if ad in ("Cora", "Citeseer", "Pubmed"):
        return Planetoid(root=kok, name=ad, transform=t)
    if ad in ("Computers", "Photo"):
        return Amazon(root=kok, name=ad, transform=t)
    if ad in ("CS", "Physics"):
        return Coauthor(root=kok, name=ad, transform=t)
    if ad == "CoraFull":
        return CoraFull(root=str(Path(kok) / "CoraFull"), transform=t)
    raise ValueError(ad)


def yukle(ad, bolme, kat, kok, bolme_dizini, dogrulamaya_izin=False):
    if ad in DOGRULAMA and not dogrulamaya_izin:
        raise RuntimeError(f"{ad} doğrulama grafı: kod dondurulmadan çalıştırılamaz.")
    ds = _yukle_ham(ad, kok)
    veri = ds[0]
    m = np.load(Path(bolme_dizini) / ad / "5_3f_85" / f"{ad.lower()}_split_{bolme}.npz", allow_pickle=True)
    katlar = m["k_fold_indices"]
    n = veri.num_nodes
    veri.train_mask = index_to_mask(torch.as_tensor(np.concatenate(np.delete(katlar, kat, axis=0))), n)
    veri.cal_mask = index_to_mask(torch.as_tensor(katlar[kat]), n)
    veri.test_mask = index_to_mask(torch.as_tensor(m["test_indices"]), n)
    veri.num_classes = int(veri.y.max()) + 1
    return veri


def yukle_sentetik(pkl_yolu):
    """DIG_storage BA_shapes / Tree_Cycles pickle'ı: [adj, x, y_tr, y_va, y_te, m_tr, m_va, m_te, kenar_etiketi]."""
    import pickle
    from torch_geometric.data import Data
    A, X, ytr, yva, yte, mtr, mva, mte, E = pickle.load(open(pkl_yolu, "rb"))
    y = torch.as_tensor((ytr + yva + yte).argmax(1))
    ei = torch.as_tensor(np.array(A.nonzero()), dtype=torch.long)
    Es = np.maximum(E, E.T)
    kenar_etiket = torch.as_tensor(Es[ei[0].numpy(), ei[1].numpy()] > 0)
    v = Data(x=torch.as_tensor(X, dtype=torch.float), edge_index=ei, y=y)
    v.train_mask, v.cal_mask, v.test_mask = (torch.as_tensor(m, dtype=torch.bool) for m in (mtr, mva, mte))
    v.num_classes = int(y.max()) + 1
    v.kenar_etiket = kenar_etiket  # edge_index sırasında
    return v
