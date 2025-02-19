# Chat with Repository

An AI-powered application that allows you to have interactive conversations with any GitHub repository's codebase. Built with Streamlit, LangChain, and Groq.

## Features

- Chat with any public GitHub repository
- Process multiple file types (.py, .js, .ts, .html, .css, .md, etc.)
- Intelligent context-aware responses using Groq's LLM models
- Vector storage using DeepLake for efficient retrieval
- Clean and intuitive Streamlit interface

## Setup

1. Clone this repository:

```bash
git clone https://github.com/yourusername/chat-with-repository.git
cd chat-with-repository
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Set up your API keys:
   - Get a Groq API key from [Groq Console](https://console.groq.com)
   - Get an ActiveLoop token from [ActiveLoop](https://www.activeloop.ai)

4. Run the application:

```bash
streamlit run main.py
```

## Usage

1. Enter your API keys in the sidebar
2. Choose between using an existing dataset or processing a new repository
3. For new repositories:
   - Enter your ActiveLoop username
   - Paste the GitHub repository URL
   - Select file extensions to process
   - Name your dataset
4. Click "Process Repository" and wait for indexing
5. Start chatting with the repository!

## Models

Supports multiple Groq models:
- Mixtral 8x7B
- DeepSeek R1 Distill LLaMA 70B
- LLaMA3 8B
- Qwen 2.5 32B

## License

MIT License