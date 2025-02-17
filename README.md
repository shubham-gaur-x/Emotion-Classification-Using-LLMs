# **Emotion Classification Using LLMs**

This project focuses on **emotion classification** using state-of-the-art **Large Language Models (LLMs)**. We train and evaluate different transformer-based models on emotion-labeled text data, aiming to classify human emotions accurately.

---

## **Objective**
The goal of this project is to analyze and classify emotions from textual data using different LLMs. The primary objectives include:
- Preprocessing and exploring the **ISEAR** emotion dataset.
- Training transformer models like **Mistral-7B** for emotion classification.
- Comparing multiple **LLMs** and analyzing their performance.
- Deploying a functional **Hugging Face** application for real-time emotion classification.

---

## **Dataset**
We use the **International Survey on Emotion Antecedents and Reactions (ISEAR)** dataset, which contains text samples labeled with emotions. The dataset is structured as follows:
- **Text:** User-generated content describing an emotional experience.
- **Emotion Label:** Predefined emotion categories (e.g., joy, anger, fear, sadness, etc.).

📂 **Dataset and model files are available in the Google Drive repository:**  
[Google Drive Repository](https://drive.google.com/drive/folders/1Qi6BfaBQwzdsnUelpqUc_WKpF_aS5Buq?usp=sharing)

---

## **Features**
- **Emotion Classification:** Classifies text into predefined emotion categories.
- **Multi-Model Evaluation:** Implements **three different LLMs** to compare performance.
- **Efficient Training:** Uses **Mistral-7B**, a powerful transformer, for optimized training.
- **Interactive Web Application:** Deploys a Hugging Face interface for real-time predictions.
- **Extensive Performance Metrics:** Evaluates models using **accuracy, F1-score, and confusion matrices**.

---

## **Demo**
🔗 **Try the emotion classifier on Hugging Face:**  
[Emotion Classifier - Hugging Face](https://huggingface.co/spaces/ShubhamGaur/emotion-classifier)

---

## **Technologies Used**
- **Transformers:** Mistral-7B, LLaMA-2, Flan-T5  
- **Libraries:** `Hugging Face Transformers`, `Datasets`, `PyTorch`, `Scikit-learn`  
- **Model Training:** Fine-tuned **Mistral-7B** for classification  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score  
- **Deployment:** Hugging Face Spaces for real-time inference  

---

## **Installation**

To set up and run the **Emotion Classification** model locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/emotion-classification.git` and navigate to the project directory with `cd emotion-classification`.
2. Create a virtual environment using `python3 -m venv env`. Activate it using `source env/bin/activate` for macOS/Linux or `env\Scripts\activate` for Windows.
3. Install all dependencies with `pip install -r requirements.txt`.
4. Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1Qi6BfaBQwzdsnUelpqUc_WKpF_aS5Buq?usp=sharing) and place it inside the `data/` directory.

---

## **Usage**

1. Train the **Mistral-7B** model for emotion classification: `python train.py`
2. Run inference on a sample text: `python predict.py --text "I am feeling very happy today!"`
3. Deploy the **Hugging Face** web app locally: `python app.py`

---

## **Model Selection & Fine-Tuning**
We experiment with three LLMs for **emotion classification**:
1. **Mistral-7B** - A highly optimized model for efficient training.
2. **LLaMA-2** - A strong baseline for text classification.
3. **Flan-T5** - Fine-tuned for instruction-based classification.

Fine-tuning is performed on **Mistral-7B** using:
- **Training Epochs:** 3
- **Batch Size:** 16
- **Optimizer:** AdamW
- **Learning Rate:** 2e-5

---

## **Evaluation & Results**
We compare models using **accuracy, precision, recall, and F1-score**.

| Model        | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|------------|--------|-----------|
| **Mistral-7B** | **87.5%** | 86.9% | 87.1% | 87.0% |
| **LLaMA-2**   | 85.2% | 84.6% | 85.0% | 84.8% |
| **Flan-T5**   | 81.4% | 80.8% | 81.0% | 80.9% |

---

## **Deployment**
The trained **Mistral-7B** model is deployed on **Hugging Face Spaces** for real-time inference.

To deploy on **Hugging Face**:
1. Push the model to **Hugging Face Model Hub**.
2. Deploy an inference API using **Gradio** or **FastAPI**.

---

## **Future Improvements**
- **Enhance model performance with RoBERTa or T5.**
- **Implement real-time emotion tracking for chatbots.**
- **Fine-tune models with multilingual emotion datasets.**

---

## **Contributors**
- **Shubham Gaur**  
📧 Email: shubham@example.com  
🔗 GitHub: [ShubhamGaur](https://github.com/ShubhamGaur)  

---

## **License**
This project is licensed under the **MIT License**.
