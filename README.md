# ✈️ Classification des Sentiments des Tweets Aériens

Ce projet vise à classifier les sentiments (positif ou négatif) exprimés dans des tweets concernant des compagnies aériennes. Plusieurs approches de deep learning sont explorées, notamment **LSTM**, **GRU avec Attention**, et **BERT (Transformers)**.

---

## 📊 Dataset

- **Source** : Twitter Airline Sentiment
- **Colonnes utilisées** : `text`, `airline_sentiment`

---

## 🧹 Prétraitement des données

- Suppression des tweets **neutres**
- Nettoyage du texte :
  - Suppression des **mentions (@user)**, **URLs**, **hashtags**, **ponctuation**, **emojis**
- Conversion des labels :
  - `0` = **négatif**, `1` = **positif**
- Équilibrage des classes par **downsampling**

---

## 🔠 Tokenization & Séquences

- **Keras Tokenizer** utilisé pour les modèles LSTM/GRU
- Padding des séquences à une **longueur fixe**

---

## 🧠 Modèles

### 🔹 LSTM
- Architecture : `Embedding → LSTM → Dropout → Dense (sigmoid)`
- Entraînement sur les séquences nettoyées

### 🔹 GRU + Attention
- Architecture : `Embedding → GRU (return_sequences=True) → Attention → Dropout → Dense (sigmoid)`
- Couche **Attention personnalisée**

### 🔹 BERT (Transformers)
- Modèle : `bert-base-uncased` via **HuggingFace Transformers**
- Tokenization et encodage adaptés à BERT
- Optimiseur avec **scheduler**

---

## 🔍 Prédiction

- Fonctions pour **prédire le sentiment** d’un tweet avec chaque modèle
- Affichage de la **probabilité associée** à la prédiction

---

## 📈 Visualisation

- Heatmap des **poids d’attention** pour le modèle GRU + Attention

---

## 💾 Sauvegarde des modèles

- Formats de sauvegarde :
  - `.h5` (HDF5)
  - `.tf` (TensorFlow SavedModel)

---

## 🧩 Dépendances principales

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `tensorflow`, `keras`
- `transformers` (HuggingFace)
- `emoji`

---

## ▶️ Exemple d’utilisation

```python
# Exemple : prédire le sentiment d’un tweet
tweet = "Le vol a été excellent, merci pour le service !"
prediction, proba = predict_with_bert(tweet)
print(f"Sentiment : {'Positif' if prediction == 1 else 'Négatif'} ({proba:.2f})")
