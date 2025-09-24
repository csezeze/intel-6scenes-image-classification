# Intel 6-Scenes Görüntü Sınıflandırma Çalışması (MobileNetV2, Keras)

## 1. Özet
Bu çalışmada, Intel 6-Scenes görüntü veri kümesi üzerinde ön-eğitimli bir evrişimsel sinir ağı mimarisi (MobileNetV2) ile aktarım öğrenmesi gerçekleştirilmiştir. Amaç, altı sahne sınıfının (Buildings, Forest, Glacier, Mountain, Sea, Street) yüksek doğrulukla ayrıştırılmasıdır. Modelin karar bölgelerini nitelendirmek üzere Grad-CAM tabanlı görselleştirme uygulanmıştır.

## 2. Veri Kümesi
- Kaynak: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
- Sınıf sayısı: 6
- Eğitim/validasyon bölünmesi `image_dataset_from_directory` ile %80/%20 oranında yapılmıştır. Test değerlendirmesi veri kümesinin “seg_test” bölümünde gerçekleştirilmiştir.

## 3. Yöntem
- Mimari: MobileNetV2 (ImageNet ağırlıkları, `include_top=False`).
- Sınıflandırıcı başlık: GlobalAveragePooling → Dropout(0.3) → Dense(6, softmax).
- Girdi boyutu: 224×224 RGB.
- Ön-işleme: `tf.keras.applications.mobilenet_v2.preprocess_input`.
- Veri artırma: `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)`.
- Optimizasyon ve düzenleme: Adam (başlangıç lr=1e−3), EarlyStopping, ReduceLROnPlateau.
- Kayıp ve metrik: `SparseCategoricalCrossentropy`, doğruluk.

## 4. Deney Düzeneği
Eğitim iki aşamada yürütülmüştür:
(i) Temel çizgi (baseline): gövde katmanları dondurulmuş, yalnızca eklenen sınıflandırıcı eğitilmiştir.
(ii) İnce ayar (fine-tuning): MobileNetV2’nin son ~20 katmanı açılmış, öğrenme oranı 1e−4’e düşürülerek ek eğitim uygulanmıştır.

## 5. Bulgular
- Test doğruluğu (baseline): **90.8%**.
- Doğrulama doğruluğu (fine-tuning, en iyi): yaklaşık %92.6 (3 epoch, son ~20 katman açık).
- Hata analizi, Glacier ile Mountain sınıfları arasında göreli olarak daha yüksek karışma bulunduğunu göstermektedir; bu durum görsel benzerlik ve ışık/tekstür koşullarından etkilenmektedir.

Ayrıntılı görseller ve raporlar:
- `assets/baseline_accuracy.png`, `assets/baseline_loss.png`
- `assets/confusion_matrix.png`
- `assets/gradcam_example.png`
- `assets/sample_preds.png`
- (opsiyonel) `assets/finetune_accuracy.png`

## 6. Grad-CAM ile Açıklanabilirlik
Grad-CAM, backbone’un son konvolüsyon katmanından elde edilen aktivasyon haritaları ile hedef sınıf skorunun gradyanlarını birleştirerek sınıf ayrımına en fazla katkı veren uzamsal bölgeleri ortaya koymaktadır. Uygulamada, konvolüsyon çıkışları ile lojitlerin aynı ileri geçişten elde edilmesine dikkat edilmiş, böylece türevler tutarlı biçimde hesaplanmıştır.

## 7. Çoğaltılabilirlik
- Ortam: Google Colab / Kaggle, TensorFlow 2.17+.
- Tohum: 42.
- Bölme: `validation_split=0.2` ve deterministik `subset` kullanımı.
- Normalizasyon ve artırma işlemleri eğitim akışına entegre edilmiştir.

## 8. Sonuç ve Gelecek Çalışmalar
Hafif ve hesaplama açısından verimli bir mimari ile yüksek sınıflandırma başarımı elde edilmiştir. Karışmaların yoğunlaştığı Glacier–Mountain çifti için daha yüksek girdi çözünürlüğü (ör. 256×256), kontrast/parlaklık düzenlemeleri ve hedefe yönelik kısa süreli ince ayarların ek iyileştirme sağlayacağı öngörülmektedir.

## 9. Lisans ve Atıf
Veri seti lisansı için Kaggle sayfasına bakınız. Kod, eğitim-öğretim amaçlı paylaşılmıştır; uygun atıfla kullanılabilir.

## Kaggle Notebook
Projenin çalışan, public sürüm: [https://www.kaggle.com/code/<kullanici>/<notebook-slug>](https://www.kaggle.com/code/zeynepozkann/notebook33bfe4ae01)

