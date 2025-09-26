🤖 AI Chatbot with Scikit-learn
A simple yet powerful chatbot built using Python and Scikit-learn that can answer questions based on a custom dataset. Features a beautiful web interface built with Streamlit.

🚀 Features
🤖 Intelligent Responses: Uses TF-IDF vectorization and cosine similarity to find the best answers

💬 Interactive Web Interface: Beautiful chat interface with real-time messaging

📊 Customizable Dataset: Easily add your own questions and answers

⚙️ Adjustable Confidence: Control how strict the chatbot is with responses

🎨 Modern UI: Dark theme with colorful message bubbles

📈 Conversation History: View your entire chat history during the session

🛠️ Technologies Used
Python - Programming language

Scikit-learn - Machine learning library

Streamlit - Web application framework

Pandas - Data manipulation

Numpy - Numerical computing

📋 Prerequisites
Before running this project, make sure you have:

Python 3.6 or higher

pip (Python package installer)

🏁 Installation
Method 1: Using requirements.txt
Clone or download the project files

bash
# Create project folder
mkdir chatbot-project
cd chatbot-project
Install dependencies

bash
pip install -r requirements.txt
Method 2: Manual installation
If you don't have a requirements.txt file, install packages individually:

bash
pip install streamlit scikit-learn pandas numpy
🎯 Usage
Navigate to the project folder

bash
cd path/to/chatbot-project
Run the application

bash
streamlit run app.py
Open your browser

The app will automatically open at http://localhost:8501

If not, manually go to the URL above

📁 Project Structure
text
chatbot-project/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
└── README.md             # This file
🎮 How to Use the Chatbot
Basic Chatting
Type your question in the input box at the bottom

Press "Send" or hit Enter

The chatbot will respond with the most relevant answer

Quick Actions
Use the quick question buttons on the right for common questions

Adjust the confidence threshold in the sidebar for more/less strict responses

Managing the Dataset
View Dataset: Check the "Show Dataset" box in the sidebar

Add New Q&A: Use the "Add New Q&A Pair" section in the sidebar

Clear Conversation: Use the "Clear Conversation" button to start fresh

⚙️ Configuration
Confidence Threshold
Lower values (0.1-0.3): Chatbot responds more often, but may be less accurate

Higher values (0.6-0.9): Chatbot responds only when very confident

Adding Custom Questions
Open the sidebar (click ">" in top-right corner if collapsed)

Scroll to "Add New Q&A Pair"

Enter your question and answer

Click "Add to Dataset"

The chatbot will immediately learn the new pair

🔧 Customization
Modifying the Dataset
Edit the create_sample_dataset() function in app.py to add your own default questions and answers:

python
def create_sample_dataset():
    data = {
        'question': [
            'your question here',
            # Add more questions...
        ],
        'answer': [
            'your answer here', 
            # Add more answers...
        ]
    }
    return pd.DataFrame(data)
Styling Changes
Modify the CSS in the st.markdown() section to change colors, fonts, and layout.

🐛 Troubleshooting
Common Issues
"Module not found" error

bash
# Reinstall missing packages
pip install streamlit scikit-learn pandas numpy
"Streamlit not recognized"

bash
# Use Python module syntax
python -m streamlit run app.py
White background/text visibility issues

The latest code includes dark theme fixes

Make sure you're using the most recent version of app.py

Port already in use

Streamlit will automatically try the next available port

Check terminal for the new URL

📊 How It Works
Machine Learning Process
Text Preprocessing: Converts text to lowercase and removes punctuation

TF-IDF Vectorization: Transforms questions into numerical vectors

Cosine Similarity: Finds the most similar question to user input

Threshold Filtering: Only responds when similarity exceeds confidence threshold

Architecture
text
User Input → Text Preprocessing → TF-IDF Vectorization → 
Cosine Similarity → Best Match → Response
🤝 Contributing
Feel free to contribute to this project by:

Adding more questions to the dataset

Improving the UI/UX

Enhancing the NLP capabilities

Adding new features

📝 License
This project is open source and available under the MIT License.

🙋‍♂️ Support
If you encounter any issues or have questions:

Check the troubleshooting section above

Ensure all dependencies are properly installed

Verify your Python version is 3.6 or higher

🎉 Success Message
Once everything is set up correctly, you should see:

text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.1.xxx:8501
Enjoy chatting with your AI assistant! 🤖💬

Happy Coding! If you like this project, please give it a ⭐ on GitHub!

