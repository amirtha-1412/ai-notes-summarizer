"""
AI Notes Summarizer - Streamlit Web Application

A beautiful web application for automated text summarization using both
extractive and abstractive techniques.
"""

import streamlit as st
from summarizer import TextSummarizer
from utils import extract_text_from_file

# Enable caching for better performance
@st.cache_resource
def get_summarizer():
    """Cache the summarizer instance to avoid reloading models."""
    return TextSummarizer()

# Page configuration
st.set_page_config(
    page_title="AI Notes Summarizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .summary-box {
        padding: 1.5rem;
        background-color: #f9fafb;
        border-left: 4px solid #3b82f6;
        border-radius: 0.5rem;
        margin: 1rem 0;
        line-height: 1.6;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    div[data-testid="stExpander"] {
        background-color: #f9fafb;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 0.5rem;
        border: 1px solid #d1d5db;
    }
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = get_summarizer()

# Header
st.markdown('<h1 class="main-header">AI Notes Summarizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform documents into concise summaries and get instant answers</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    
    # Input method selection
    input_method = st.radio(
        "How do you want to provide text?",
        ["Type or Paste Text", "Upload a File"],
        help="Choose your preferred input method"
    )
    
    st.divider()
    
    # Simple summarization mode selection
    st.subheader("Summarization Mode")
    
    summary_mode = st.radio(
        "Choose what works best for you:",
        [
            "Quick Summary (Recommended)",
            "Academic/Research",
            "Professional/Business",
            "Advanced (Custom Settings)"
        ],
        help="Select based on your use case"
    )
    
    # Set defaults based on mode
    if summary_mode == "Quick Summary (Recommended)":
        st.info(" Fast processing\nn Balanced quality\n Best for general use")
        technique = "Extractive"
        extractive_algorithm = "textrank"
        sentence_count = 5
        abstractive_model = "sshleifer/distilbart-cnn-12-6"
        max_length = 130
        min_length = 30
        
    elif summary_mode == "Academic/Research":
        st.info(" Preserves exact quotes\n Key findings highlighted\n Citation-friendly")
        technique = "Extractive"
        extractive_algorithm = "lsa"
        sentence_count = st.slider("Summary Length", 3, 10, 7, help="Number of key sentences")
        abstractive_model = "facebook/bart-large-cnn"
        max_length = 150
        min_length = 50
        
    elif summary_mode == "Professional/Business":
        st.info(" Natural language\n Executive-ready\n Concise & clear")
        technique = "Abstractive"
        extractive_algorithm = "textrank"
        sentence_count = 5
        abstractive_model = "facebook/bart-large-cnn"
        max_length = st.slider("Summary Length", 50, 200, 100, help="Approximate word count")
        min_length = 40
        
    else:  # Advanced
        st.warning(" Advanced mode - customize all settings")
        
        technique = st.selectbox(
            "Technique",
            ["Extractive", "Abstractive", "Both"],
            help="Extractive: Selects key sentences. Abstractive: Rewrites in new words."
        )
        
        if technique in ["Extractive", "Both"]:
            with st.expander(" Extractive Options"):
                extractive_algorithm = st.selectbox(
                    "Algorithm",
                    ["textrank", "lsa", "lexrank", "luhn"],
                    format_func=lambda x: {
                        "textrank": "TextRank (Balanced)",
                        "lsa": "LSA (Academic)",
                        "lexrank": "LexRank (News)",
                        "luhn": "Luhn (Technical)"
                    }[x]
                )
                sentence_count = st.slider("Sentences", 3, 15, 5)
        
        if technique in ["Abstractive", "Both"]:
            with st.expander(" Abstractive Options"):
                abstractive_model = st.selectbox(
                    "Model",
                    [
                        "facebook/bart-large-cnn",
                        "sshleifer/distilbart-cnn-12-6",
                        "google/pegasus-xsum"
                    ],
                    format_func=lambda x: {
                        "facebook/bart-large-cnn": "BART (High Quality)",
                        "sshleifer/distilbart-cnn-12-6": "DistilBART (Fast)",
                        "google/pegasus-xsum": "Pegasus (Ultra Short)"
                    }[x]
                )
                max_length = st.slider("Max Length", 50, 300, 150)
                min_length = st.slider("Min Length", 30, 100, 50)
    
    st.divider()
    st.markdown("### Tips")
    st.markdown("""
    - **Quick Summary**: Best for most users
    - **Academic**: Use for research papers
    - **Professional**: For reports & emails
    - **Advanced**: Full control over settings
    """)
    
    st.divider()
    
    # Help section
    with st.expander("Help & Tips"):
        st.markdown("""
        **Quick Start:**
        1. Choose input method (text or file)
        2. Select summarization mode
        3. Generate summary
        4. Ask questions about your document
        
        **Supported Formats:**
        - PDF, DOCX, TXT files
        - Direct text input
        
        **Best Practices:**
        - Provide at least 100 words for better results
        - Use Academic mode for research papers
        - Use Professional mode for business documents
        """)

# Initialize input_text variable
input_text = None

# Main content area
if input_method == "Type or Paste Text":
    text_input = st.text_area(
        "Enter your text here:",
        height=300,
        placeholder="Paste or type the text you want to summarize...\n\nTip: The more text you provide, the better the summary will be!",
        help="Enter at least 50 characters"
    )
    input_text = text_input
    
elif input_method == "Upload a File":
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, Word (DOCX), Text (TXT)"
    )
    
    if uploaded_file:
        try:
            with st.spinner(" Extracting text from your file..."):
                file_bytes = uploaded_file.read()
                input_text = extract_text_from_file(file_bytes, uploaded_file.name)
                st.success(f"Successfully extracted text from **{uploaded_file.name}**")
                
                # Show preview
                with st.expander("Preview Extracted Text"):
                    st.text_area("Extracted Text", input_text, height=200, disabled=True)
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            input_text = None
    else:
        input_text = None

# Example text option
with st.expander("Need sample text? Click here"):
    st.markdown("Use this example to test the summarizer:")
    example_text = """Artificial intelligence (AI) is transforming the way we live and work. Machine learning, a subset of AI, enables computers to learn from data without being explicitly programmed. Deep learning, which uses neural networks with multiple layers, has achieved remarkable success in areas like image recognition, natural language processing, and game playing. AI applications are now ubiquitous, from virtual assistants like Siri and Alexa to recommendation systems on Netflix and Amazon. In healthcare, AI is being used to diagnose diseases, discover new drugs, and personalize treatment plans. Autonomous vehicles rely on AI to navigate roads safely. However, the rapid advancement of AI also raises important ethical questions about privacy, job displacement, and algorithmic bias. As AI continues to evolve, it's crucial that we develop it responsibly, ensuring that its benefits are widely shared and its risks are carefully managed."""
    if st.button("Use Example Text", key="use_example"):
        st.session_state.example_used = True
        st.rerun()

if 'example_used' in st.session_state and st.session_state.example_used:
    input_text = """Artificial intelligence (AI) is transforming the way we live and work. Machine learning, a subset of AI, enables computers to learn from data without being explicitly programmed. Deep learning, which uses neural networks with multiple layers, has achieved remarkable success in areas like image recognition, natural language processing, and game playing. AI applications are now ubiquitous, from virtual assistants like Siri and Alexa to recommendation systems on Netflix and Amazon. In healthcare, AI is being used to diagnose diseases, discover new drugs, and personalize treatment plans. Autonomous vehicles rely on AI to navigate roads safely. However, the rapid advancement of AI also raises important ethical questions about privacy, job displacement, and algorithmic bias. As AI continues to evolve, it's crucial that we develop it responsibly, ensuring that its benefits are widely shared and its risks are carefully managed."""
    st.session_state.example_used = False

# Summarize button
if st.button("Generate Summary", type="primary", use_container_width=True):
    if not input_text or len(input_text.strip()) < 50:
        st.warning(" Please provide at least 50 characters of text to summarize.\n\nTip: Use the example text above to try it out!")
    else:
        try:
            with st.spinner(" Analyzing your text and generating summary... Please wait."):
                summarizer = st.session_state.summarizer
                
                if technique == "Extractive":
                    summary, metadata = summarizer.extractive_summary(
                        input_text,
                        sentence_count=sentence_count,
                        algorithm=extractive_algorithm
                    )
                    
                    # Display results
                    st.success(" Summary generated successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(
                            f'<div class="stats-box"><h3>{metadata["original_stats"]["word_count"]:,}</h3><p>Original Words</p></div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f'<div class="stats-box"><h3>{metadata["summary_stats"]["word_count"]:,}</h3><p>Summary Words</p></div>',
                            unsafe_allow_html=True
                        )
                    with col3:
                        st.markdown(
                            f'<div class="stats-box"><h3>{metadata["compression_ratio"]}%</h3><p>Compression</p></div>',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("### Extractive Summary")
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                    st.caption(f"Algorithm: {metadata['algorithm']}")
                    
                elif technique == "Abstractive":
                    summary, metadata = summarizer.abstractive_summary(
                        input_text,
                        max_length=max_length,
                        min_length=min_length,
                        model_name=abstractive_model
                    )
                    
                    # Display results
                    st.success(" Summary generated successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(
                            f'<div class="stats-box"><h3>{metadata["original_stats"]["word_count"]:,}</h3><p>Original Words</p></div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f'<div class="stats-box"><h3>{metadata["summary_stats"]["word_count"]:,}</h3><p>Summary Words</p></div>',
                            unsafe_allow_html=True
                        )
                    with col3:
                        st.markdown(
                            f'<div class="stats-box"><h3>{metadata["compression_ratio"]}%</h3><p>Compression</p></div>',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("### Abstractive Summary")
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                    st.caption(f"Model: {metadata['model']}")
                    
                else:  # Both
                    summaries, metadata = summarizer.hybrid_summary(
                        input_text,
                        extractive_sentences=sentence_count,
                        extractive_algorithm=extractive_algorithm,
                        abstractive_max_length=max_length,
                        abstractive_min_length=min_length,
                        abstractive_model=abstractive_model
                    )
                    
                    # Display results
                    st.success(" Summaries generated successfully!")
                    
                    # Original stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(
                            f'<div class="stats-box"><h3>{metadata["original_stats"]["word_count"]:,}</h3><p>Original Words</p></div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f'<div class="stats-box"><h3>{metadata["extractive"]["summary_stats"]["word_count"]:,}</h3><p>Extractive Words</p></div>',
                            unsafe_allow_html=True
                        )
                    with col3:
                        st.markdown(
                            f'<div class="stats-box"><h3>{metadata["abstractive"]["summary_stats"]["word_count"]:,}</h3><p>Abstractive Words</p></div>',
                            unsafe_allow_html=True
                        )
                    
                    # Extractive summary
                    st.markdown("### Extractive Summary")
                    st.markdown(f'<div class="summary-box">{summaries["extractive"]}</div>', unsafe_allow_html=True)
                    st.caption(f"Algorithm: {metadata['extractive']['algorithm']} | Compression: {metadata['extractive']['compression_ratio']}%")
                    
                    # Abstractive summary
                    st.markdown("### Abstractive Summary")
                    st.markdown(f'<div class="summary-box">{summaries["abstractive"]}</div>', unsafe_allow_html=True)
                    st.caption(f"Model: {metadata['abstractive']['model']} | Compression: {metadata['abstractive']['compression_ratio']}%")
                
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            st.exception(e)

# Q&A Section
st.markdown("---")
st.subheader("Ask Questions About Your Document")
st.markdown("Get instant answers from your text using extractive or abstractive methods.")

# Check if we have text available (from any source)
# Priority: 1. Current input_text, 2. Example text flag, 3. Session state

# First, check if user just used example text
if 'example_used' in st.session_state and st.session_state.example_used:
    example_text = """Artificial intelligence (AI) is transforming the way we live and work. Machine learning, a subset of AI, enables computers to learn from data without being explicitly programmed. Deep learning, which uses neural networks with multiple layers, has achieved remarkable success in areas like image recognition, natural language processing, and game playing. AI applications are now ubiquitous, from virtual assistants like Siri and Alexa to recommendation systems on Netflix and Amazon. In healthcare, AI is being used to diagnose diseases, discover new drugs, and personalize treatment plans. Autonomous vehicles rely on AI to navigate roads safely. However, the rapid advancement of AI also raises important ethical questions about privacy, job displacement, and algorithmic bias. As AI continues to evolve, it's crucial that we develop it responsibly, ensuring that its benefits are widely shared and its risks are carefully managed."""
    st.session_state.document_text = example_text
    st.session_state.has_document = True

# Check if we have input_text from text area or file upload
if input_text and len(input_text.strip()) >= 50:
    st.session_state.document_text = input_text
    st.session_state.has_document = True

# Now check if we have any document in session state
if st.session_state.get('has_document') and 'document_text' in st.session_state:
    document_text = st.session_state.document_text
    
    # Show that document is loaded
    st.success(f"Document loaded ({len(document_text)} characters). You can now ask questions!")
    
    # Question input
    user_question = st.text_input(
        "Your Question:",
        placeholder="e.g., What are the main points? Who is mentioned? What is the conclusion?",
        help="Ask any question about the document",
        key="qa_question"
    )
    
    # Answer method selection
    col1, col2 = st.columns([3, 1])
    with col1:
        qa_method = st.radio(
            "Answer Method:",
            ["Extractive Only (Fast)", "Abstractive (Multiple Sentences)", "Both Methods"],
            horizontal=False,
            help="All methods are lightweight with no downloads required!",
            key="qa_method"
        )
    
    # Show info about methods
    if "Abstractive" in qa_method or "Both" in qa_method:
        st.info("Abstractive method uses TF-IDF similarity to find and combine multiple relevant sentences for more complete answers.")
    
    # Ask button
    if st.button("Get Answer", type="primary", use_container_width=True, key="qa_button"):
        if not user_question or len(user_question.strip()) < 3:
            st.warning("Please enter a question (at least 3 characters).")
        else:
            try:
                with st.spinner("Finding the answer..."):
                    summarizer = st.session_state.summarizer
                    
                    if "Extractive Only" in qa_method:
                        answer, metadata = summarizer.answer_question_extractive(
                            document_text,
                            user_question,
                            context_sentences=3
                        )
                        
                        st.success("Answer found!")
                        st.markdown("### Extractive Answer")
                        st.info(answer)
                        st.caption(f"Method: {metadata['method']} | Sentences: {metadata['sentences_found']} | Confidence: {metadata['confidence']:.2%}")
                        
                    elif "Abstractive" in qa_method:
                        answer, metadata = summarizer.answer_question_abstractive(
                            document_text,
                            user_question
                        )
                        
                        st.success("Answer generated!")
                        st.markdown("### Abstractive Answer")
                        st.info(answer)
                        st.caption(f"Method: {metadata['method']} | Model: {metadata.get('model', 'N/A')} | Confidence: {metadata['confidence']:.2%}")
                        
                    else:  # Both
                        answers, metadata = summarizer.answer_question_hybrid(
                            document_text,
                            user_question
                        )
                        
                        st.success("Answers generated!")
                        
                        # Display both answers in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Extractive Answer")
                            st.markdown(f'<div class="summary-box">{answers["extractive"]}</div>', unsafe_allow_html=True)
                            st.caption(f"Sentences: {metadata['extractive']['sentences_found']} | Confidence: {metadata['extractive']['confidence']:.2%}")
                        
                        with col2:
                            st.markdown("### Abstractive Answer")
                            st.markdown(f'<div class="summary-box">{answers["abstractive"]}</div>', unsafe_allow_html=True)
                            st.caption(f"Method: {metadata['abstractive'].get('model', 'N/A')} | Confidence: {metadata['abstractive']['confidence']:.2%}")
                        
                        st.info("**Tip**: Extractive answers use single sentences, while Abstractive answers combine multiple sentences for more complete responses.")
                        
            except Exception as e:
                st.error(f" Error answering question: {str(e)}")
                st.exception(e)
    
    # Example questions
    with st.expander("Example Questions & Tips"):
        st.markdown("""
        **Try asking questions like:**
        - What is this document about?
        - What are the main points?
        - Who or what is mentioned?
        - What is the conclusion?
        - What are the key findings?
        - When did this happen?
        - Why is this important?
        
        **Pro Tip:** Start with "Extractive Only (Fast)" for instant results! 
        AI methods require downloading a large model on first use.
        """)
        
else:
    st.info("Please provide text or upload a document above (and optionally generate a summary first) to use the Q&A feature.")
    st.markdown("**Steps to get started:**")
    st.markdown("1. Click 'Need sample text?' above and use the example")
    st.markdown("2. OR type/paste your own text")
    st.markdown("3. OR upload a PDF/DOCX/TXT file")
    st.markdown("4. Then scroll down here to ask questions!")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #9ca3af; font-size: 0.875rem;">Built with Streamlit, Transformers & Sumy</p>',
    unsafe_allow_html=True
)
