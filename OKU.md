# BAP 18840 — deney kodu (hafta 2)

## İçerik

- `bapkanit/veri.py` — Cora/Pubmed/Computers (+ doğrulama grafları, kilitli) ve GATS'ın açık bölmeleri; BA-Shapes yükleyici.
- `bapkanit/omurga.py` — GATS ile aynı GCN ve eğitim düzeni.
- `bapkanit/kalibrasyon.py` — TS ve ISTS (11 betimleyici, CV + bir-standart-hata kuralı). `ek` sütunlarla H3 sinyalleri eklenecek.
- `bapkanit/olcut.py` — NLL, Brier, ECE (15 kutu), AURC.
- `bapkanit/gsat_dugum.py` — düğüm düzeyi GSAT.
- `cora_kontrol.py` — GCN + TS + ISTS, Cora, 15 çalıştırma.
- `gsat_kontrol.py` — GSAT: BA-Shapes açıklama AUC'si ve Cora doğruluğu.

## Kurulum

Hafta 1'deki ortam yeterli (`conda activate bap`). Ek olarak iki veri deposu gerekiyor (kod klasörünün bir üstüne):

```
git clone --depth 1 https://github.com/hans66hsu/GATS.git
git clone --depth 1 https://github.com/kimiyoung/planetoid.git
git clone --depth 1 https://github.com/divelab/DIG_storage.git
mkdir veri\Cora\raw
copy planetoid\data\ind.cora.* veri\Cora\raw\
```

Klasör düzeni: `GATS/`, `planetoid/`, `DIG_storage/`, `veri/`, `kod/` yan yana.

## Çalıştırma

```
cd kod
python cora_kontrol.py
python gsat_kontrol.py
```

Bulut ortamındaki sonuçlar `cora_kontrol.log` ve `gsat_kontrol.log` içinde.
