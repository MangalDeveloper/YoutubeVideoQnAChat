# 🎥 YouTube Video QnA Chat (RAG-powered with LangChain)

This project is an **interactive chatbot** that allows you to query any YouTube video using its transcript. It is powered by **RAG (Retrieval-Augmented Generation)** and **LangChain**, enabling the assistant to answer only from the video’s content.  

---

## 🚀 Features
- 🔍 **Automatic Transcript Fetching** – Extracts captions using [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/).  
- ✂️ **Chunking & Embeddings** – Transcript split into manageable chunks and embedded with `OpenAIEmbeddings`.  
- 🧠 **RAG Workflow** – Questions are answered using retrieved chunks (not hallucinated).  
- 💬 **Chat-based UI** – Powered by **Streamlit’s chat interface** with sequential history.  
- ⚡ **Streaming AI Responses** – Answers appear **token by token** for a real-time feel.  
- ❌ **Exit Command** – Type `exit` to reset and load a new YouTube video.  
- 🎨 **Custom UI** – Dark blue gradient background with white text for readability.  

---

## 🛠️ How It Works (RAG + LangChain Behind the Scenes)

### 🔹 Step 1: Transcript Retrieval  
The transcript is fetched from YouTube using **YouTubeTranscriptApi**. If captions are disabled, the app stops gracefully.  

### 🔹 Step 2: Text Splitting  
The transcript (which can be thousands of words long) is divided into **overlapping chunks** (1000 characters, 200 overlap) using LangChain’s `RecursiveCharacterTextSplitter`.  

### 🔹 Step 3: Embeddings & Vector Store  
- Each chunk is embedded into a **vector representation** using `OpenAIEmbeddings (text-embedding-3-small)`.  
- These embeddings are stored in **FAISS**, a fast similarity search library.  

### 🔹 Step 4: Retrieval  
When the user asks a question, the system:  
1. Embeds the query.  
2. Finds the **top-k (4)** most relevant transcript chunks using FAISS similarity search.  

### 🔹 Step 5: Prompt Construction  
A **LangChain PromptTemplate** ensures the model only uses transcript context:  

You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.


### 🔹 Step 6: Answer Generation  
- Retrieved context + question → sent to **ChatOpenAI (gpt-4o-mini)**.  
- Model streams its response back to the user, displayed in real time.  

### 🔹 Step 7: Stateful Chat  
- Chat history is stored in `st.session_state`.  
- If the user types `exit`, chat resets and prompts for a new video.  

---

💻 Tech Stack

    Frontend/UI: Streamlit
    LLM: OpenAI GPT-4o-mini
    RAG Components: LangChain
    Vector DB: FAISS
    Embeddings: OpenAI text-embedding-3-small
    Transcript API: youtube-transcript-api

📊 Example Workflow

    User enters a YouTube video link → App fetches transcript.
    Transcript is chunked & embedded → stored in FAISS.
    User asks: “What was explained about machine learning in this video?”
    FAISS retrieves top transcript chunks.
    LLM answers:
    "The speaker explained that machine learning allows systems to learn from data without explicit programming, highlighting supervised and unsupervised learning."

⚠️ Limitations

    ❌ Only works if video has captions available.
    ❌ Model won’t answer outside transcript scope (by design).
    💰 Requires an OpenAI API key (usage cost applies).


🚀 Future Improvements

    Add multi-language support for transcripts.
    Store multiple videos in one session for cross-video QnA.
    Support offline embeddings (e.g., HuggingFace models).
    Add summarization mode (short summary of video).

## 📸 App Screenshot  

![App Screenshot](./Screenshots/streamlit_app.png)
