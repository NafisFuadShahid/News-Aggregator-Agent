import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

import google.generativeai as genai

# --- Configure Gemini ---
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Gemini API key not set. Please add GEMINI_API_KEY to your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
# Use gemini-1.5-pro for better reasoning
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# ----- UI -----
st.title("NewsBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Add configuration options
st.sidebar.subheader("Configuration")
chunk_size = st.sidebar.slider("Chunk Size", 300, 1000, 600)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 200, 100)
num_sources = st.sidebar.slider("Number of Sources to Use", 3, 8, 5)

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_bert.pkl"
main_placeholder = st.empty()

# ----- Better Embeddings -----
# Using a more recent and better model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

if process_url_clicked:
    try:
        # Filter out empty URLs
        valid_urls = [u.strip() for u in urls if u.strip()]
        if not valid_urls:
            st.error("Please provide at least one valid URL.")
            st.stop()
            
        loader = UnstructuredURLLoader(urls=valid_urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        
        if not data:
            st.error("No data could be loaded from the provided URLs.")
            st.stop()

        # Better text splitting strategy
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '. ', '! ', '? ', ', '],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        
        if not docs:
            st.error("No documents could be created from the loaded data.")
            st.stop()

        vectorstore_bert = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_bert, f)
        
        main_placeholder.text(f"Processing completed! Created {len(docs)} document chunks ✅")

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")

query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            # Use configurable number of sources
            docs = vectorstore.similarity_search(query, k=num_sources)
            
            # Also get similarity scores to filter low-relevance results
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=num_sources)
            
            # Filter out documents with very low similarity (high distance)
            relevant_docs = [doc for doc, score in docs_with_scores if score < 1.5]
            
            if not relevant_docs:
                st.warning("No highly relevant information found in the processed articles for your question.")
                relevant_docs = [doc for doc, score in docs_with_scores[:3]]  # Use top 3 anyway
            
            context = "\n\n".join([f"Source {i+1}:\n{doc.page_content}" 
                                 for i, doc in enumerate(relevant_docs)])

            # Improved prompt with better instructions
            prompt = f"""You are an expert news analyst. Your task is to answer the user's question using ONLY the information provided in the news articles below.

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain enough information to fully answer the question, clearly state what information is missing
3. Cite specific sources when making claims (e.g., "According to Source 1...")
4. If there are conflicting information between sources, mention this
5. Be precise and factual - avoid speculation
6. If you cannot answer the question based on the provided context, say so clearly

CONTEXT FROM NEWS ARTICLES:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer based on the available information:"""
            
            try:
                # Configure generation parameters for better quality
                generation_config = genai.types.GenerationConfig(
                    temperature=0.1,  # Lower temperature for more factual responses
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1000,
                )
                
                response = gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                answer = response.text
                
            except Exception as e:
                answer = f"Error from Gemini: {e}"
                st.error("There was an error generating the response. Please try again.")

            st.header("Answer")
            st.write(answer)
            
            # Show relevance scores
            if docs_with_scores:
                st.subheader("Sources Used (with relevance scores):")
                for i, (doc, score) in enumerate(docs_with_scores[:len(relevant_docs)], 1):
                    source_url = doc.metadata.get('source', 'URL unknown')
                    relevance = "High" if score < 0.8 else "Medium" if score < 1.2 else "Low"
                    st.write(f"**Source {i}** (Relevance: {relevance}): {source_url}")
                    
                    # Show a snippet of the content
                    with st.expander(f"Preview Source {i}"):
                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        
            # Debug information
            with st.expander("Debug Information"):
                st.write(f"Total chunks processed: {len(docs) if 'docs' in locals() else 'N/A'}")
                st.write(f"Chunks used for answer: {len(relevant_docs)}")
                st.write(f"Query length: {len(query)} characters")
                
    else:
        st.warning("Please process URLs first before asking questions.")
        
# Add tips section
with st.sidebar.expander("💡 Tips for Better Results"):
    st.write("""
    **For better answers:**
    - Use specific, clear questions
    - Ensure URLs contain relevant content
    - Try different chunk sizes if results are poor
    - Increase number of sources for complex questions
    
    **Chunk Size Guide:**
    - Small (300-400): Better for specific facts
    - Medium (500-700): Balanced approach
    - Large (800-1000): Better for context-heavy questions
    """)