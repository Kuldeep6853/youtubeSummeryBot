import streamlit as st
import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader


load_dotenv()


# -------- functions ----------

def get_youtube_id(url: str):
    if "watch?v=" in url:
        return url.split("watch?v=")[-1]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1]
    return url


def get_youtube_transcript(video_url: str):
    try:
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
        docs = loader.load()
        transcript = "\n".join([doc.page_content for doc in docs])
        return transcript

    except Exception as e:
        return f"❌ Error: {str(e)}"


def build_rag(transcript_text):
    documents = [Document(page_content=transcript_text)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts = [doc.page_content for doc in chunks]
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8}
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant that answers questions based ONLY on the transcript.

Transcript:
-----------
{context}

Question: {question}
""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    return parallel_chain | prompt | llm | parser



# ---------- UI ----------

st.title("🎥 YouTube RAG Assistant")
st.write("Paste a YouTube link, ask a question, and get AI-generated answers based on the transcript.")


youtube_url = st.text_input("🔗 Enter YouTube Video URL")

question = st.text_input("❓ Ask Your Question")


if st.button("Generate Answer"):

    if not youtube_url.strip() or not question.strip():
        st.warning("Please provide both YouTube URL and a question.")
        st.stop()

    with st.spinner("Fetching transcript and thinking... ⏳"):
        transcript = get_youtube_transcript(youtube_url)

        if transcript.startswith("❌"):
            st.error(transcript)
        else:
            rag_chain = build_rag(transcript)
            answer = rag_chain.invoke(question)

            st.success("Answer Ready 🧠")
            st.text_area("📌 Result:", value=answer, height=500)
