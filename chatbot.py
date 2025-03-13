import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r", encoding='utf-8') as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

# Custom CSS for the UI
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f7fc;
            color: #333;
        }
        .sidebar .sidebar-content {
            background-color: #1E2952;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff; 
            border-radius: 10px;
            border: 1px solid #1E2952;
            padding: 10px;
            color: #000000;
            caret-color: #0f1116; 
        }
        .stTextArea>div>div>textarea {
            background-color: #ffffff; 
            border-radius: 15px;
            padding: 10px; 
            font-size: 16px; 
            min-height: 70px; 
            color: #000000; 
            caret-color: #090446; /* Cursor color */
        }
        .stTextArea>div>div>textarea[disabled] {
            background-color: #1B1B1B; 
            color: rgba(255, 255, 255, 1); 
            border-radius: 15px;
            padding: 10px; 
            font-size: 16px; 
        }
        .stButton>button {
            background-color: #1E2952;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1E2952;
        }
        .stText {
            font-size: 16px;
            color: #333;
        }
        .stTitle {
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)



def main():
    global counter
    st.title(" Chatbot using NLP")
    
    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}", placeholder="Type your message here...")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}", disabled=True)

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to develop a chatbot that understands and responds to user input based on intents. The chatbot utilizes the Natural Language Processing (NLP) library and Logistic Regression to extract intents and entities from user input. It is built using Streamlit, a Python library for creating interactive web applications.")

        st.subheader("Project Overview:")

        st.write("""
      The project is divided into two main parts:
    1. NLP Techniques and Logistic Regression: These are used to train the chatbot on labeled intents and entities.
    2. Chatbot Interface: Built using the Streamlit web framework, this interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset for this project consists of labeled intents and entities stored in a list. 
        - Intents: Represent the user's purpose (e.g., "greeting", "news", "joke").
        - Entities: Extracted components from the user input (e.g., "Hi", "Tell me the latest news", "Tell me a joke").
        - Text: The raw user input.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface, built with Streamlit, features a text input box for user queries and a chat window to display responses. It utilizes a trained model to generate relevant replies based on user input.")

        st.subheader("Conclusion:")

        st.write("This project features a chatbot designed to understand and respond to user input based on intents. It leverages NLP and Logistic Regression for training, with an interactive interface built using Streamlit. Future enhancements can include expanding the dataset, incorporating advanced NLP techniques, and integrating deep learning models.")

if __name__ == '__main__':
    main()


 