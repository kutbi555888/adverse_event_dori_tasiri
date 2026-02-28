# 💊 FAERS25Q4 — Adverse Event (Nojo‘ya ta’sir) Multilabel Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-8A2BE2?style=for-the-badge)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-red?style=for-the-badge)

> 🧠 **Maqsad:** FAERS (FDA Adverse Event Reporting System) 2025 Q4 ma’lumotlari asosida, matndan (reaksiya/symptom) **bir vaqtning o‘zida bir nechta nojo‘ya ta’sir kategoriyalarini** (multilabel) bashorat qilish.

---

## 📌 Project nomi
**FAERS25Q4**

---

## 🧭 Project nima haqidaligi?
Bu loyiha FAERS hisobotlaridan olingan **symptom/reaksiya matni** (MedDRA PT termlar) asosida:

✅ qaysi organ/tizim guruhiga mansub nojo‘ya ta’sirlar borligini  
✅ bir vaqtning o‘zida bir nechta label bilan  
**multilabel classification** ko‘rinishida bashorat qiladi.

---

## 🎯 Project yo‘nalishi
- **Machine Learning**
- **NLP (Text Classification)**
- **Multilabel Classification**
- **Explainable AI (SHAP)**

---

## 🔮 Project nimani predict qiladi?
Model kiruvchi matndan quyidagi kabi label’larni bashorat qiladi (misol):

- `cardiovascular`
- `gastrointestinal`
- `respiratory`
- `renal`
- `dermatologic`
- `psychiatric`
- `edema_swelling`
- `hypersensitivity_allergy`
- `infections`
- `pain_general`
- … (jami 21 ta label)

Ya’ni: **bitta report → bir nechta label** bo‘lishi mumkin.

---

## 🧾 Data manbasi va data haqida
📂 **Manba:** FAERS 25Q4 (DEMO/DRUG/REAC/INDI/OUTC/THER/RPSR/DELETE) TXT fayllari  
📌 Avval TXT fayllar o‘qildi, kerakli ustunlar ajratildi va merge qilindi.

Projectda yakuniy ishlatilgan dataset:
- `Data/Raw_data/faers_25Q4_targets_multilabel_v2.csv`

---

## 🎯 Target (label) nima?
Target — bu **multilabel** bo‘lib, har bir report uchun bir nechta label 1 bo‘lishi mumkin.

### ✅ Target qanday ko‘rinishda?
- CSV ichida `y_labels` ko‘rinishida:
  - `cardiovascular; edema_swelling; hypersensitivity_allergy`
- Ichki train uchun esa:
  - `y_<label>` ko‘rinishida 0/1 ustunlar (21 ta ustun)

---

## 🧩 Multilabel degani nima?
📌 Oddiy classification’da 1 ta target bo‘ladi:
- `Class = A yoki B`

📌 Multilabel’da esa:
- `Class = A + C + F` (bir vaqtning o‘zida bir nechta)

Bu loyiha **multilabel**: bir reportda bir nechta adverse event kategoriyasi bo‘lishi mumkin.

---

## 🧠 Feature’lar (model kirishlari)
### 1) Asosiy feature — Text
- `REAC_pt_symptom_v2` (yoki `REAC_pt_symptom`)

Matn ko‘rinishi:

Sinus tachycardia; Generalised oedema; Cardiac arrest; Hypotension; ...


### 2) Textdan chiqarilgan NLP feature’lar
📌 **FeatureUnion** orqali:
- **Word TF-IDF**: `ngram_range=(1,2)`
- **Char TF-IDF (char_wb)**: `ngram_range=(3,5)`
- **Meta feature’lar** (matndan):
  - `log1p_len`
  - `n_terms`
  - `n_uniq_terms`

### 3) Feature Selection
- `chi2` asosida feature tanlash (mask):
  - `feature_selector.joblib` (mask/selected_idx)

---

## 🛠️ Projectda qilingan ishlar tartibi (pipeline)
Quyidagi ketma-ketlikda ish bajarildi:

### 01) Data loading
📥 FAERS TXT fayllarni o‘qish: DEMO/DRUG/REAC/INDI/…  
🔗 Merge va cleaning

### 02) Target creation
🎯 `y_labels` va `y_<label>` ustunlarini yaratish  
📌 Multilabel mapping

### 03) Split
📌 Train / Validation / Test split

### 04) Feature engineering (NLP)
🧪 TF-IDF (word + char) + meta features  
✅ `tfidf_vectorizer.joblib` sifatida saqlandi

### 05) Feature selection
✂️ Chi-square (chi2) orqali feature mask  
✅ `feature_selector.joblib` sifatida saqlandi

### 06) Baseline training
⚙️ Baseline modellari:
- LogisticRegression (OvR)
- LinearSVC
- SGD (log_loss / hinge)
- Calibrated SVC

### 07) Improvement training
🔧 Improved variantlar va threshold tuning

### 08) Hyperparameter tuning (Optuna)
🧠 Optuna bilan tuning (NO_OVERSAMPLING)

### 09) Best model selection
🏆 Eng yaxshi model tanlandi va saqlandi:
- `Models/best_model/optuna_logreg_best/`

### 10) Compare results
📊 Baseline vs Improvement vs Tuning vs Best Model taqqoslandi

### 11) Offline testing
🧪 CSV’dan real primaryid olib:
- TRUE vs PRED
- hard-case mining (qiyin case’lar)

### 12) Explainability (SHAP)
🔍 Global + Local SHAP:
- global shap bar
- signed contribution bar (+/-)
- summary plots
- local waterfall/bar/force

---

## ✅ Project maqsadi
🎯 FAERS reportlarini avtomatik analiz qilib, **qaysi tizimlarda adverse event bo‘lish ehtimoli borligini** tezda aniqlash.

Bu:
- farmacovigilance (dori xavfsizligi) ishlarini tezlashtirish
- signal detection / risk analysis
- medical text classification

uchun foydali.

---

## 📦 Papkalar tuzilmasi (asosiy)
```text
Data/
  Raw_data/
    faers_25Q4_targets_multilabel_v2.csv
  Engineered_data/
    fe_v1/
      tfidf_vectorizer.joblib
      meta.json
  Feature_Selected/
    fe_v1_fs_chi2_v1/
      feature_selector.joblib
      X_train.npz, X_test.npz
      Y_train.npy, Y_test.npy

Models/
  best_model/
    optuna_logreg_best/
      optuna_logreg_best.joblib
      optuna_logreg_best_thresholds.json

visuals/
  SHAP/
    shap_summary.png
    global_shap_bar.png
    global_shap_bar_positive_class.png
    shap_signed_bar_topN.png
    local_shap_waterfall_idx_0.png
    local_shap_bar_idx_0.png
    local_shap_force_idx_0.html

results/
  compare/
  offline/




  🚀 Qanday ishlatiladi? (quickstart)
1) Environment
pip install -r requirements.txt
2) Offline predict (bitta primaryid)

1 ta record uchun:

CSV’dan primaryid topiladi

model predict qiladi

# new_object (minimal)
new_object = {
  "primaryid": 260447931,
  "REAC_pt_symptom_v2": "Sinus tachycardia; Generalised oedema; Cardiac arrest; ..."
}
📌 Natijalar (high level)

🏆 Best model:

micro_f1 ≈ 0.978

macro_f1 ≈ 0.966

🧾 Project avtori

👤 Muallif: Qutbiddin
🧩 Yo‘nalish: ML / NLP / Pharmacovigilance

🌟 Qo‘shimcha

🧠 Multilabel threshold ishlatilgan (har label uchun alohida threshold)

🔍 Hard-case miner orqali qiyin recordlar topilgan

📊 Compare results: baseline/improvement/tuning/best_model

🧷 Sticker zone 😄

🩺💊🧬📊🧠🔍🧪🚀✅🏆🔥