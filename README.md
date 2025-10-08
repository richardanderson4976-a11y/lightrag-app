# 🔦 LightRAG with Gemini

A powerful RAG (Retrieval-Augmented Generation) application built with LightRAG and Google's Gemini AI.

## Features

- 📄 **Multi-format Support**: Upload TXT, MD, PDF, and DOCX files
- 💬 **Interactive Chat**: Ask questions about your documents
- 🧠 **Advanced RAG**: Uses LightRAG's knowledge graph approach
- 🚀 **Fast & Free**: Powered by Gemini 2.0 Flash
- 🎨 **Clean UI**: Built with Streamlit

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/lightrag-app.git
cd lightrag-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Get Your API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key and paste it in the app sidebar

## Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click "New app"
4. Select your forked repository
5. Add your Gemini API key in Secrets:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

## Query Modes

- **Hybrid**: Best overall results (recommended)
- **Local**: Focus on specific entities and relationships
- **Global**: High-level summaries across documents
- **Naive**: Simple vector similarity search

## Tech Stack

- **Frontend**: Streamlit
- **RAG Engine**: LightRAG
- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: text-embedding-004

## License

MIT License
