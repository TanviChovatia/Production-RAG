# 📄 Data Folder

This folder is used to store PDF documents for the RAG system.

## 📥 What to put here
- Research papers
- Technical documents
- Any PDF files you want to query

## 📂 Example
data/
└── bert paper.pdf

## 🔧 Where it is used
- CLI queries (`main.py`)
- Evaluation (`evals/run_eval.py`)

## ⚠️ Important Notes
- Only PDF files are supported
- File names can contain spaces, but avoid special characters
- Large files may increase processing time

## 💡 Tip
- Add domain-specific documents (e.g., AI, healthcare, finance) for better results
- More relevant documents = better answers

## 🌐 Streamlit UI
- The UI also supports direct PDF uploads
- Files uploaded via UI are processed temporarily and do not need to be stored here