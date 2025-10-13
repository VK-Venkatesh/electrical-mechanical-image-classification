# 🔍 Streamlit Image Similarity Search

An interactive **Streamlit** web app that finds visually similar images using **deep learning (MobileNetV2)** embeddings and **cosine similarity**.

---

## 🚀 Features
- Upload or select an image to find similar ones from a preloaded dataset.
- Uses **MobileNetV2** pretrained on ImageNet for feature extraction.
- Caches image embeddings for fast inference.
- Simple and interactive UI built with **Streamlit**.

---

## 🧠 How It Works
1. Each image in the dataset is passed through MobileNetV2 (without the classification layer).
2. The extracted 1280-dimensional feature vector is **normalized** and saved to `features.npy`.
3. When you query an image, its feature vector is compared to the cached dataset using **cosine similarity**.
4. The app returns the **Top N most similar images**.

---
