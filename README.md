# Goruntu isleme-ile-Hava-Savunma-Unsuru-Tespit-Sistemi
Görüntü işleme ve derin öğrenme tabanlı hava savunma unsurları tespit ve sınıflandırma sistemi.

Görüntü İşleme Tabanlı Hava Savunma Tehdit Tespit ve Sınıflandırma Sistemi

Bu repository, bitirme projesi kapsamında geliştirilen ve sahadaki kritik hava savunma unsurlarını yüksek doğrulukla tespit edip sınıflandıran derin öğrenme tabanlı bir karar destek sistemine aittir.

## 📌 Proje Mimarisi ve Çalışma Mantığı
Sistem, gerçek dünya operasyonel şartlarına uygun olarak **iki aşamalı (Cascaded) bir karar mekanizması** ile çalışır:
1. **Binary Sınıflandırma (Var / Yok):** Sahnede herhangi bir tehdit unsuru (`Fire Based Radar` veya `Launcher`) olup olmadığını denetler. Arka planı güvenli bölge olarak filtreler.
2. **Spesifik Sınıflandırma:** Sahneye giren tehdidin türünü (`Fire Based Radar` veya `Launcher`) net bir şekilde ayrıştırır.

## 📊 Başarım Analizi ve Metrikler
Modelin **102 adet test görseli** üzerinden elde ettiği operasyonel başarım sonuçları:

* **Genel Doğruluk (Accuracy):** %83.33
* **Tehdit Yakalama Oranı (Recall):** %82.14 *(Sahadaki tehditleri kaçırmama hassasiyeti)*
* **Güvenilirlik Oranı (Precision):** %86.79 *(Yanlış alarm üretmeme kararlılığı)*

## 🗂️ Veri Seti Yapısı
* **Taban Veri Seti:** Modelin başlangıç eğitimi için Roboflow Universe üzerinden temin edilen açık kaynaklı topluluk verileri kullanılmıştır.
* **Özgün Genişletme:** Veri setinde ben ilave olarak Askeri kaynaklardan Uydu görüntülerinden ve Sivil cekimlerden de acık kaynaklı olarak görüntüler elde ettim ve bunu projenin veri setine ekledim.

---
## 📂 Repository İçeriği

* `main.py`: Modelin ana çıkarım ve karar mekanizmasının çalıştırıldığı script.
* `train.py`: Modelin eğitim süreçlerinin, hiperparametre ayarlarının ve veri seti entegrasyonunun yönetildiği script.
* `Confusion_Matrix_With_Metrics.png`: Test seti sonuçlarının detaylı metriklerle (`Accuracy`, `Recall`, `Precision`) görselleştirildiği performans matrisi.
