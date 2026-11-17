"""Sonradan kalibrasyon: sıcaklık ölçekleme (TS) ve ISTS (Huang, arXiv 2611.00175).

ISTS: T_i = softplus(b + w . phi_i) + 0.05, phi standartlaştırılmış betimleyiciler.
Ceza lambda * ||w||^2, lambda kalibrasyon kümesi içinde 5 katlı CV + bir-standart-hata kuralıyla seçilir.
H3 testi için `ek` ile betimleyici matrisine yeni sütunlar (kanıt sinyalleri) eklenebilir.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, to_undirected

TAU_MIN = 0.05
LAMBDA_IZGARA = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
BETIMLEYICI_ADLARI = ["guven", "entropi", "komsu_uyumu", "goreli_guven", "komsu_benzerligi",
                      "log_derece", "LP_uyumu", "LP_guveni", "uzaklik_1", "uzaklik_2", "uzaklik_3+"]


# ---------------- betimleyiciler ----------------
def _komsu_ortalama(deger, ei, n):
    s, d = ei
    top = torch.zeros(n, *deger.shape[1:]).index_add_(0, d, deger[s])
    der = degree(d, n).clamp(min=1).view(-1, *([1] * (deger.dim() - 1)))
    return top / der


def atlama_uzakligi(ei, egitim_maske, n, en_fazla=3):
    """Her düğümün en yakın eğitim düğümüne atlama uzaklığı (en_fazla'da kırpılır)."""
    uz = torch.full((n,), en_fazla, dtype=torch.long)
    sinir = egitim_maske.clone()
    uz[sinir] = 0
    ziyaret = sinir.clone()
    for h in range(1, en_fazla):
        yeni = torch.zeros(n, dtype=torch.bool)
        yeni[ei[1][sinir[ei[0]]]] = True
        yeni &= ~ziyaret
        uz[yeni] = h
        ziyaret |= yeni
        sinir = yeni
    return uz


def etiket_yayilimi(ei, y, egitim_maske, n, K, alfa=0.9, L=10):
    s, d = ei
    der = degree(d, n)
    dinv = der.pow(-0.5); dinv[torch.isinf(dinv)] = 0
    w = dinv[s] * dinv[d]
    Y = torch.zeros(n, K); Y[egitim_maske] = F.one_hot(y[egitim_maske], K).float()
    Q = Y.clone()
    for _ in range(L):
        Q = alfa * torch.zeros(n, K).index_add_(0, d, Q[s] * w.view(-1, 1)) + (1 - alfa) * Y
    return Q / Q.sum(1, keepdim=True).clamp(min=1e-12)


def betimleyiciler(logits, ei, y, egitim_maske):
    n, K = logits.shape
    ei = to_undirected(ei, num_nodes=n)
    p = logits.softmax(1)
    c, yhat = p.max(1)
    H = -(p * p.clamp(min=1e-12).log()).sum(1) / math.log(K)
    s, d = ei
    der = degree(d, n)
    uyum = torch.zeros(n).index_add_(0, d, (yhat[s] == yhat[d]).float()) / der.clamp(min=1)
    goreli = c - _komsu_ortalama(c.view(-1, 1), ei, n)[:, 0]
    cos = F.cosine_similarity(p[s], p[d], dim=1)
    benzer = torch.zeros(n).index_add_(0, d, cos) / der.clamp(min=1)
    Q = etiket_yayilimi(ei, y, egitim_maske, n, K)
    lp_uyum = Q[torch.arange(n), yhat]
    lp_guv = Q.max(1).values
    uz = atlama_uzakligi(ei, egitim_maske, n)
    phi = torch.stack([c, H, uyum, goreli, benzer, torch.log1p(der), lp_uyum, lp_guv,
                       (uz == 1).float(), (uz == 2).float(), (uz >= 3).float()], 1)
    return phi


def standartlastir(phi):
    mu, sd = phi.mean(0), phi.std(0)
    return (phi - mu) / sd.clamp(min=1e-8)


# ---------------- uydurma ----------------
def _nll_dugum(logits, y, T):
    return F.cross_entropy(logits / T.view(-1, 1), y, reduction="none")


def ts_uydur(logits, y):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def kapat():
        opt.zero_grad()
        k = _nll_dugum(logits, y, logT.exp().expand(len(y))).mean(); k.backward(); return k
    opt.step(kapat)
    return logT.exp().item()


def _ists_uydur(logits, y, phi, lam, T0):
    b0 = math.log(math.expm1(max(T0 - TAU_MIN, 1e-4)))
    b = torch.tensor([b0], requires_grad=True)
    w = torch.zeros(phi.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([b, w], lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def kapat():
        opt.zero_grad()
        T = F.softplus(b + phi @ w) + TAU_MIN
        k = _nll_dugum(logits, y, T).mean() + lam * (w ** 2).sum(); k.backward(); return k
    opt.step(kapat)
    return b.detach(), w.detach()


def ists_sicaklik(b, w, phi):
    return F.softplus(b + phi @ w) + TAU_MIN


def ists_uydur(logits_cal, y_cal, phi_cal, tohum=0, kat=5):
    """lambda'yı CV + bir-standart-hata kuralıyla seçer, tüm kalibrasyon kümesinde yeniden uydurur."""
    n = len(y_cal)
    g = torch.Generator().manual_seed(tohum)
    sira = torch.randperm(n, generator=g)
    kat_id = torch.empty(n, dtype=torch.long); kat_id[sira] = torch.arange(n) % kat
    kayip = np.zeros((len(LAMBDA_IZGARA), n))
    for j, lam in enumerate(LAMBDA_IZGARA):
        for f in range(kat):
            tr, te = kat_id != f, kat_id == f
            T0 = ts_uydur(logits_cal[tr], y_cal[tr])
            b, w = _ists_uydur(logits_cal[tr], y_cal[tr], phi_cal[tr], lam, T0)
            T = ists_sicaklik(b, w, phi_cal[te])
            kayip[j, te.numpy()] = _nll_dugum(logits_cal[te], y_cal[te], T).numpy()
    ort = kayip.mean(1)
    j_en = int(ort.argmin())
    secilen = j_en
    for j in range(len(LAMBDA_IZGARA)):
        delta = kayip[j] - kayip[j_en]
        if delta.mean() <= delta.std(ddof=1) / math.sqrt(n):
            secilen = max(secilen, j)
    lam = LAMBDA_IZGARA[secilen]
    T0 = ts_uydur(logits_cal, y_cal)
    b, w = _ists_uydur(logits_cal, y_cal, phi_cal, lam, T0)
    return dict(b=b, w=w, lam=lam)
