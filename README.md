# 🧠 Text Summarizer & Q&A Assistant

This script summarizes long texts and allows you to ask questions about the generated summary. It is built using Hugging Face models (`facebook/bart-large-cnn` for summarization and `deepset/roberta-base-squad2` for Q&A).

---

## ⚙️ Features

- ✅ Summarizes input text into short, medium, or long formats  
- ✅ Enhances the summary using an additional refinement model  
- ✅ Allows asking questions based on the summary  
- ✅ Works locally with GPU acceleration (CUDA-supported)

---

## 🧩 Technologies Used

- 🤗 Transformers (`pipeline`)  
- 🤖 LangChain (`HuggingFacePipeline`, `PromptTemplate`)  
- 💪 PyTorch with CUDA support  

---

## 🚀 How to Run Locally

1. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Install PyTorch with CUDA support:**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

3. **Install other dependencies:**

   ```bash
   pip install transformers langchain langchain-huggingface
   ```

4. **Run the script:**

   ```bash
   python main.py
   ```

   (Replace `main.py` with your script’s filename)

---

## 💡 Example Usage

```text
Enter text to summarize:
> Artificial Intelligence (AI) is rapidly transforming...

Enter the length (short/medium/long):
> short

**Generated summary:**
AI is changing industries quickly...

Ask a question about the summary (or type 'exit' to stop):
> What is AI doing?

**Answer:**
changing industries
```

---

## 📌 Notes

- The script uses GPU by default (`device=0`). Make sure you have a CUDA-compatible GPU. If not, change `device=0` to `device=-1`.
- The models are downloaded from Hugging Face, so an internet connection is required on the first run.
