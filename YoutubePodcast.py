import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re

# Load API key from .env
load_dotenv()

# ---- Streamlit UI ----
st.set_page_config(page_title="YouTube Video QnA Chat", page_icon="🎥", layout="centered")

# 🔹 Add gradient dark blue background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #001f3f, #003366);
        color: white;
    }
    .stChatMessage {
        color: white !important;
    }
    .stMarkdown, .stMarkdown p {
        color: white !important;
    }
    /* Only make the label (not the input text) white */
    .stTextInput label {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎥 YouTube Video QnA Chat")

# Function to extract YouTube video ID
def extract_video_id(url_or_id):
    pattern = r"(?:v=|youtu\.be/|youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})"
    match = re.search(pattern, url_or_id)
    return match.group(1) if match else url_or_id.strip()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "main_chain" not in st.session_state:
    st.session_state.main_chain = None
if "last_video_id" not in st.session_state:
    st.session_state.youtube_id = None

# Reset function
def reset_chat():
    st.session_state.chat_history = []
    st.session_state.main_chain = None
    st.session_state.youtube_id = None

# If no active video processing, show input for YouTube ID
if st.session_state.main_chain is None:
    video_input = st.text_input("Enter YouTube Video ID:", placeholder="e.g. Gfr50f6ZBvo")

    if video_input:
        video_id = extract_video_id(video_input)
        st.session_state.youtube_id = video_id
        st.session_state.chat_history = []

        with st.spinner("📜 Fetching transcript..."):
            try:
                ytt_api = YouTubeTranscriptApi()
                fetched = ytt_api.fetch(video_id, languages=["en"])
                raw_transcript = fetched.to_raw_data()
                transcript = " ".join(entry["text"] for entry in raw_transcript)

            except TranscriptsDisabled:
                st.error("❌ No captions available for this video.")
                st.stop()

        with st.spinner("🔍 Processing transcript into vector store..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                streaming=True
            )

            prompt = PromptTemplate(
                template="""
                  You are a helpful assistant.
                  Answer ONLY from the provided transcript context.
                  If the context is insufficient, just say you don't know.

                  {context}
                  Question: {question}
                """,
                input_variables=["context", "question"]
            )

            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            parser = StrOutputParser()
            st.session_state.main_chain = parallel_chain | prompt | llm | parser

        st.success("✅ Video is ready for Q&A!")

# --- Chat Section ---
if st.session_state.main_chain is not None:
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message("user" if role == "You" else "assistant"):
            st.markdown(message)

    # Chat input field (auto clears)
    user_question = st.chat_input("💬 Ask a question (type 'exit' to end)")

    if user_question:
        if user_question.lower().strip() == "exit":
            reset_chat()
            st.rerun()  # 🔹 Forces immediate UI refresh

        # Show user message instantly
        st.session_state.chat_history.append(("You", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        # Stream AI answer
        streamed_answer = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in st.session_state.main_chain.stream(user_question):
                streamed_answer += chunk
                placeholder.markdown(streamed_answer)

        # Save AI response
        st.session_state.chat_history.append(("AI", streamed_answer))
