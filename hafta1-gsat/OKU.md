# Hafta 1 — GSAT'ı kendi makinende çalıştırma

Amaç: resmi GSAT kodunun BA-2Motifs üzerinde makaledeki açıklama başarısını (ROC AUC 98.74) verdiğini kendi ekran kartında görmek.

Bulut ortamında (yalnız işlemci, 3 tohum) elde edilen sonuç: 0.989, 0.978, 0.993; ortalama 0.987.

## Kurulum (Windows, Anaconda Prompt)

```
conda create -n bap python=3.11 -y
conda activate bap
```

PyTorch'u ekran kartı desteğiyle kur: https://pytorch.org/get-started/locally/ sayfasındaki seçiciden (Windows, Pip, CUDA) çıkan komutu çalıştır. Sonra:

```
pip install torch_geometric pyyaml scikit-learn tqdm tensorboard matplotlib networkx rdkit ogb
git clone https://github.com/Graph-COM/GSAT.git
cd GSAT
git apply ..\gsat_modern_env.patch
cd src
python run_gsat.py --dataset ba_2motifs --backbone GIN --cuda 0
```

(`gsat_modern_env.patch` dosyasını GSAT klasörünün bir üstüne koy.)

Kontrol: `python -c "import torch; print(torch.cuda.is_available())"` True vermeli.

## Yama ne değiştiriyor

Resmi kod 2021 sürümlerine (torch 1.10, PyG 2.0) göre yazılmış. Güncel sürümlerde çalışması için:
- torch_sparse ve torch_scatter bağımlılıkları kaldırıldı (PyG'nin kendi işlevleri kullanılıyor).
- Mesaj geçirme katmanlarında `propagate_type` satırlarına `edge_atten` eklendi.
- `torch.load` çağrılarına `weights_only=False` eklendi.
- Görselleştirmede kaldırılmış bir matplotlib çağrısı değiştirildi.
Modelin matematiği değişmedi.

## Beklenen çıktı

Her tohumun sonunda şu satır: `Best Test X AUROC: 0.9xx`. 10 tohum, ekran kartında birkaç dakika. Ortalama 0.97'nin altındaysa bana haber ver.
