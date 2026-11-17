"""Değerlendirme ölçütleri. Girdi: log-olasılıklar [n, K] ve etiketler [n]."""
import torch


def nll(logp, y):
    return -logp[torch.arange(len(y)), y].mean().item()


def brier(logp, y):
    p = logp.exp()
    oh = torch.nn.functional.one_hot(y, p.shape[1]).float()
    return ((p - oh) ** 2).sum(1).mean().item()


def ece(logp, y, kutu=15):
    p = logp.exp()
    guv, tah = p.max(1)
    dogru = (tah == y).float()
    sinir = torch.linspace(0, 1, kutu + 1)
    e = torch.zeros(())
    for a, b in zip(sinir[:-1], sinir[1:]):
        m = (guv > a) & (guv <= b)
        if m.any():
            e += m.float().mean() * (guv[m].mean() - dogru[m].mean()).abs()
    return e.item()


def aurc(skor, hata):
    """Risk-kapsama eğrisi altındaki alan. skor: yüksek = daha güvenli. hata: 1 = yanlış."""
    sira = torch.argsort(skor, descending=True)
    h = hata[sira].float()
    risk = h.cumsum(0) / torch.arange(1, len(h) + 1)
    return risk.mean().item()


def hepsi(logp, y):
    tah = logp.argmax(1)
    return dict(acc=(tah == y).float().mean().item(), nll=nll(logp, y), brier=brier(logp, y),
                ece=ece(logp, y), aurc=aurc(logp.max(1).values, tah != y))
