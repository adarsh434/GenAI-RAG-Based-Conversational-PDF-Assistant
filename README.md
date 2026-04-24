# 📄 GenAI RAG-Based Conversational PDF Assistant

> An AI-powered document assistant that lets you **chat with your PDFs** using Retrieval-Augmented Generation (RAG) — built with LangChain, Chroma, and Streamlit.

---

## 🚀 Features

- 📤 **Upload any PDF** and start asking questions instantly
- 🧠 **RAG pipeline** — retrieves relevant context before generating answers
- 💬 **Conversational interface** — ask follow-up questions naturally
- ⚡ **Fast vector search** using Chroma
- 🎨 **Clean Streamlit UI** with a custom-styled chat input
- 🔒 Runs locally — your documents stay on your machine

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | ChatMistralAI |
| Embeddings | MistralAIEmbeddings|
| Vector Store | Chroma |
| RAG Framework | LangChain |
| PDF Parsing | PyPDF2 |

---

## 📁 Project Structure

```
GenAI-RAG-PDF-Assistant/
│
├── app.py                  # Main Streamlit app
├── RAG_Test.py             # RAG chain setup (retriever + LLM)
├── create_database.py      # PDF loading and chunking
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/GenAI-RAG-PDF-Assistant.git
cd GenAI-RAG-PDF-Assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Add your API keys to .env
```

**.env file:**
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🧩 How It Works

```
User uploads PDF
       │
       ▼
PDF is parsed & split into chunks
       │
       ▼
Chunks are embedded & stored in FAISS vector store
       │
       ▼
User asks a question
       │
       ▼
Relevant chunks are retrieved via similarity search
       │
       ▼
LLM generates answer using retrieved context (RAG)
       │
       ▼
Answer displayed in chat UI
```

---

## 📸 Screenshots

![Demo Screenshot](Upload.png)
![Demo Screenshot](Chat1.png)
![Demo Screenshot](chat2.png)
![Demo Screenshot](chat3.png)
![Demo Screenshot](chat4.png)

---

## 📦 Requirements

```
streamlit
langchain
langchain-openai
faiss-cpu
pypdf2
python-dotenv
openai
```
