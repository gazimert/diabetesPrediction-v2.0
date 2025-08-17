# 🩺 Diyabet Tahmin Modeli (MLP + Flask API)

Bu proje, diyabet hastalığını tahmin etmek için **MLP (Multi-Layer
Perceptron)** algoritmasını kullanır. Veri seti önişleme adımlarından
geçirilir, **SMOTE ile dengesizlik sorunları giderilir**, model eğitilir
ve farklı metrikler ile değerlendirilir. Ayrıca, modeli gerçek zamanlı tahminler yapabilen bir Flask API'si ile kullanıma sunar.

## 🔹 Kullanılan Teknolojiler

-   Python\
-   Pandas, NumPy (Veri işleme)\
-   Scikit-learn (Önişleme, train/test split, metrikler)\
-   imbalanced-learn (SMOTE ile oversampling)\
-   TensorFlow / Keras (MLP modeli)\
-   Matplotlib & Seaborn (Grafikler, görselleştirme)\
-   Joblib (Scaler kaydetme / yükleme)
-   Flask (REST API)

------------------------------------------------------------------------

## 🌐 Flask API Kullanımı

API'yi başlatmak için:

``` bash
python diabetes_flask.py
```

API varsayılan olarak `http://0.0.0.0:5001` üzerinde çalışır.

------------------------------------------------------------------------

## ⚙️ 1. Veri Hazırlama Adımları

1.  Veri seti yüklenir (`diabetes_prediction_dataset.csv`).\
2.  Bağımlı değişken (`diabetes`) ve bağımsız değişkenler ayrılır.\
3.  Kategorik değişkenler **One-Hot Encoding** ile dönüştürülür.\
4.  **StandardScaler** ile veriler normalize edilir.\
5.  Eğitim ve test setlerine %80 - %20 oranında bölünür.\
6.  Eğitim setine **SMOTE** uygulanarak sınıf dengesizliği azaltılır.

------------------------------------------------------------------------

## 🤖 2. MLP Modeli

-   **1. Gizli Katman:** 32 nöron, ReLU aktivasyonu\
-   **2. Gizli Katman:** 16 nöron, ReLU aktivasyonu\
-   **Çıkış Katmanı:** 1 nöron, Sigmoid aktivasyonu\
-   **Optimizatör:** Adam (lr=0.001)\
-   **Kayıp Fonksiyonu:** Binary Crossentropy\
-   **Metrik:** Accuracy

### Overfitting Önlemleri

-   `EarlyStopping` ile erken durdurma\
-   `class_weight` ile dengesiz sınıf ağırlıklarının dengelenmesi

------------------------------------------------------------------------

## 📊 3. Model Eğitimi

Model, eğitim seti üzerinde 20 epoch boyunca eğitilir.\
Doğrulama setinde performans takip edilir.

Eğitim sürecinde:\
- Eğitim / Doğrulama kayıpları grafiği\
- Eğitim / Doğrulama doğruluk grafiği\
- Karışıklık Matrisi ve Heatmap görselleştirilir.

------------------------------------------------------------------------

## ✅ 4. Model Değerlendirme

-   Accuracy, Precision, Recall, F1-Score raporlanır.\
-   Confusion Matrix görselleştirilir.

### Örnek Sonuç:

``` text
Accuracy: 0.82
Precision: 0.80
Recall: 0.78
F1-Score: 0.79
```

------------------------------------------------------------------------

## 🧪 5. Yeni Hasta Tahmini

Yeni bir hastanın verileri JSON formatında girildiğinde:\
1. One-Hot Encoding ile dönüştürülür.\
2. Eksik sütunlar sıfır ile doldurulur.\
3. Eğitim seti ile aynı sütun sırasına göre hizalanır.\
4. `StandardScaler` ile normalize edilir.\
5. Model tahmin olasılığı verir.

Örnek çıktı:

``` text
Tahmin edilen olasılık: 0.72
✅ Bu kişi diyabet hastası olabilir (1).
```

### 📡 Tahmin Yapma (API Kullanımı)

#### Endpoint

**`POST /predict`**

#### Örnek JSON Girdi:

``` json
{
    "age": 59,
    "hypertension": 0,
    "heart_disease": 1,
    "bmi": 35.31,
    "HbA1c_level": 6.7,
    "blood_glucose_level": 180,
    "gender_Female": 1,
    "gender_Male": 0,
    "gender_Other": 0,
    "smoking_history_never": 1,
    "smoking_history_current": 0,
    "smoking_history_ever": 0,
    "smoking_history_former": 0,
    "smoking_history_not current": 0,
    "smoking_history_No Info": 0
}
```

#### Örnek Çıktı:

``` json
{
  "prediction": 1,
  "probability": 0.76
}
```

-   `prediction`: 0 (sağlıklı) veya 1 (diyabet)\
-   `probability`: Tahmin olasılığı (0.0--1.0)

------------------------------------------------------------------------

## 🚀 6. Çalıştırma

``` bash
# Gerekli kütüphaneleri yükle
pip install pandas numpy scikit-learn imbalanced-learn tensorflow matplotlib seaborn joblib
# API’yi başlat
python diabetes_flask.py
```

Eğitim sonrası:\
- `mlp_model.h5` → Eğitilmiş model\
- `scaler.pkl` → Normalizasyon parametreleri

oluşturulur.

------------------------------------------------------------------------

## 📌 Notlar

-   API'ye gönderilen veriler one-hot encoded sütunları (örn. gender_Female) içermeli ve eğitimde kullanılan sütun sırasıyla tam olarak aynı olmalıdır.
