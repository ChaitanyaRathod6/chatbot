import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

# Set page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FIXED CSS - Better styling with visible text
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #0e1117;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #262730;
    }
    .user-message {
        background-color: #1f77b4;
        color: white;
        padding: 12px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: right;
        margin-left: 80px;
        border-bottom-right-radius: 5px;
    }
    .bot-message {
        background-color: #2ecc71;
        color: white;
        padding: 12px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: left;
        margin-right: 80px;
        border-bottom-left-radius: 5px;
    }
    .stTextInput>div>div>input {
        color: black;
        background-color: white;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1668a3;
    }
    /* Fix sidebar text color */
    .css-1d391kg, .css-1d391kg p {
        color: white !important;
    }
    /* Fix main area text color */
    .main .block-container {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class SimpleChatbot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.questions = []
        self.answers = []
        self.X = None
        
    def train(self, questions, answers):
        """Train the chatbot with questions and answers"""
        self.questions = questions
        self.answers = answers
        
        # Preprocess and vectorize the questions
        processed_questions = [self.preprocess_text(q) for q in questions]
        self.X = self.vectorizer.fit_transform(processed_questions)
        return f"Chatbot trained with {len(questions)} Q&A pairs"
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text
    
    def get_response(self, user_input, threshold=0.3):
        """Get response for user input"""
        if self.X is None:
            return "Chatbot not trained yet. Please train with data first."
        
        # Preprocess user input
        processed_input = self.preprocess_text(user_input)
        
        # Vectorize user input
        input_vector = self.vectorizer.transform([processed_input])
        
        # Calculate similarity with all questions
        similarities = cosine_similarity(input_vector, self.X)
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[0, best_match_idx]
        
        # Return answer if similarity is above threshold
        if best_similarity > threshold:
            return self.answers[best_match_idx]
        else:
            return "I'm not sure how to answer that. Can you try rephrasing your question?"
    
    def add_qna(self, question, answer):
        """Add new Q&A pair to the chatbot"""
        self.questions.append(question)
        self.answers.append(answer)
        
        # Retrain with new data
        processed_questions = [self.preprocess_text(q) for q in self.questions]
        self.X = self.vectorizer.fit_transform(processed_questions)
        return f"Added new Q&A pair. Total pairs: {len(self.questions)}"

# Sample dataset
def create_sample_dataset():
    data = {
        'question': [
            'hello', 'hi', 'how are you', 'what is your name', 'who are you',
            'what can you do', 'tell me a joke', 'what is machine learning',
            'how does python work', 'what is artificial intelligence', 'goodbye',
            'bye', 'see you later', 'thanks', 'thank you', 'what time is it',
            'how old are you', 'where are you from', 'what is your purpose',
            'do you like music', 'what is python', 'how to learn programming'
        ],
        'answer': [
            'Hello! How can I help you today?',
            'Hi there! What can I do for you?',
            'I am just a chatbot, but I am functioning well! How about you?',
            'I am a chatbot created to help answer your questions!',
            'I am an AI chatbot designed to assist with information and conversations.',
            'I can answer questions, have conversations, and provide information on various topics.',
            'Why did the chatbot cross the road? To get to the other website!',
            'Machine learning is a subset of AI that allows computers to learn without being explicitly programmed.',
            'Python is an interpreted programming language that executes code line by line.',
            'Artificial intelligence is the simulation of human intelligence in machines.',
            'Goodbye! Have a great day!',
            'Bye! Come back if you have more questions!',
            'See you later! Take care!',
            'You are welcome! Happy to help!',
            'My pleasure! Let me know if you need anything else.',
            'I am a chatbot, I do not have access to real-time clock information.',
            'I was just created, so I am brand new!',
            'I exist in the digital world to help users like you!',
            'My purpose is to assist you with information and answer your questions.',
            'I do not have personal preferences, but I can talk about music!',
            'Python is a popular programming language known for its simplicity and readability.',
            'Start with basic concepts, practice regularly, and build small projects to improve.'
        ]
    }
    return pd.DataFrame(data)

# Initialize chatbot
@st.cache_resource
def initialize_chatbot():
    chatbot = SimpleChatbot()
    df = create_sample_dataset()
    chatbot.train(df['question'].tolist(), df['answer'].tolist())
    return chatbot, df

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot, st.session_state.dataset = initialize_chatbot()

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.3

# Main app
def main():
    st.markdown('<h1 class="main-header">🤖 AI Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Chatbot Settings")
        
        # Confidence threshold slider
        st.session_state.threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Higher values make the chatbot more conservative in answering"
        )
        
        st.subheader("Dataset Information")
        st.write(f"Total Q&A pairs: {len(st.session_state.dataset)}")
        
        # Show dataset
        if st.checkbox("Show Dataset"):
            st.dataframe(st.session_state.dataset)
        
        # Add new Q&A pair
        st.subheader("Add New Q&A Pair")
        with st.form("add_qna"):
            new_question = st.text_input("New Question")
            new_answer = st.text_area("Answer")
            submitted = st.form_submit_button("Add to Dataset")
            if submitted and new_question and new_answer:
                result = st.session_state.chatbot.add_qna(new_question, new_answer)
                st.success(result)
                # Update dataset in session state
                st.session_state.dataset = pd.DataFrame({
                    'question': st.session_state.chatbot.questions,
                    'answer': st.session_state.chatbot.answers
                })
        
        # Clear conversation
        if st.button("Clear Conversation"):
            st.session_state.conversation = []
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat Interface")
        
        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display conversation
        for message in st.session_state.conversation:
            if message['type'] == 'user':
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["text"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message"><strong>Bot:</strong> {message["text"]}</div>', 
                           unsafe_allow_html=True)
        
        # If no conversation yet, show welcome message
        if not st.session_state.conversation:
            st.markdown('<div class="bot-message"><strong>Bot:</strong> Hello! I\'m your AI assistant. How can I help you today?</div>', 
                       unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # User input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Type your message here:", key="user_input")
            send_button = st.form_submit_button("Send")
            
            if send_button and user_input:
                # Add user message to conversation
                st.session_state.conversation.append({
                    'type': 'user',
                    'text': user_input,
                    'time': time.time()
                })
                
                # Get bot response
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.get_response(
                        user_input, 
                        st.session_state.threshold
                    )
                
                # Add bot response to conversation
                st.session_state.conversation.append({
                    'type': 'bot',
                    'text': response,
                    'time': time.time()
                })
                
                st.rerun()
    
    with col2:
        st.subheader("Quick Actions")
        
        # Quick questions buttons
        st.write("Try these quick questions:")
        quick_questions = [
            "Hello", 
            "What is your name?", 
            "Tell me a joke", 
            "What is machine learning?",
            "How are you?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"btn_{question}"):
                # Add user message to conversation
                st.session_state.conversation.append({
                    'type': 'user',
                    'text': question,
                    'time': time.time()
                })
                
                # Get bot response
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.get_response(
                        question, 
                        st.session_state.threshold
                    )
                
                # Add bot response to conversation
                st.session_state.conversation.append({
                    'type': 'bot',
                    'text': response,
                    'time': time.time()
                })
                
                st.rerun()
        
        # Bot information
        st.subheader("Bot Info")
        st.info("""
        This chatbot uses:
        - **Scikit-learn** for machine learning
        - **TF-IDF** for text vectorization
        - **Cosine similarity** for matching questions
        """)
        
        # Conversation stats
        if st.session_state.conversation:
            user_msgs = len([m for m in st.session_state.conversation if m['type'] == 'user'])
            bot_msgs = len([m for m in st.session_state.conversation if m['type'] == 'bot'])
            st.write(f"**Conversation Stats:**")
            st.write(f"User messages: {user_msgs}")
            st.write(f"Bot responses: {bot_msgs}")

# Run the app
if __name__ == "__main__":
    main()