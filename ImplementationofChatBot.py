#  Implementation of Chatbotg Using NPL 


#Importing neccesary Libraries
import nltk
import random
import os
import ssl
import streamlit as s
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath('nltk_data'))
nltk.download('punkt') 

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "What's up?", "How are you?", "Hey there"],
        "responses": ["Hey there!", "Hello! How can I assist you?", "Hi! What's on your mind?", "Hey! Hope you're having a great day!"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care", "Catch you later"],
        "responses": ["Goodbye! Have a great day!", "See you later!", "Take care!", "Until next time!"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "I appreciate it", "Much appreciated"],
        "responses": ["You're very welcome!", "No problem at all!", "Happy to help!", "Glad I could assist you!"]
    },
    {
        "tag": "about",
        "patterns": ["Who are you?", "What can you do?", "Tell me about yourself", "What's your purpose?"],
        "responses": ["I'm a chatbot designed to assist you!", "I'm here to answer your questions and help you with various topics.", "Think of me as your virtual assistant, always ready to help!"]
    },
    {
        "tag": "help",
        "patterns": ["I need help", "Can you help me?", "What should I do?", "Help me"],
        "responses": ["Of course! What do you need help with?", "I'm here to assist. Let me know what you need!", "How can I make things easier for you today?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you?", "What's your age?", "When were you created?"],
        "responses": ["I don’t age like humans, but I’m as fresh as the latest update!", "I exist in the digital world, so time is just a concept to me!", "Let's just say I was born in the cloud!"]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like?", "How's the weather today?", "Is it raining outside?"],
        "responses": ["I can't check real-time weather, but you can try a weather app!", "For accurate weather updates, I recommend checking a weather website!", "I wish I could tell you, but checking a weather app might be better!"]
    },
    {
        "tag": "joke",
        "patterns": ["Tell me a joke", "Make me laugh", "Say something funny"],
        "responses": ["Why don’t skeletons fight each other? They don’t have the guts!", "I told my laptop I needed a break, and now it won’t stop sending me vacation ads!", "Parallel lines have so much in common. It’s a shame they’ll never meet."]
    },
    {
        "tag": "motivation",
        "patterns": ["Motivate me", "I need motivation", "Give me some inspiration"],
        "responses": ["The best time to start was yesterday. The next best time is now!", "You are capable of amazing things. Keep going!", "Success starts with the decision to try."]
    },
    {
        "tag": "time",
        "patterns": ["What time is it?", "Can you tell me the time?", "What's the current time?"],
        "responses": ["I can’t check real-time clocks, but your device should have the answer!", "Time is precious, don’t waste it! Check your clock and make the most of your day!", "Your watch might be better at this, but I’m sure it’s a great time to do something productive!"]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget?", "What's a good budgeting strategy?", "How do I create a budget?"],
        "responses": ["A simple way to budget is using the 50/30/20 rule: 50% for needs, 30% for wants, and 20% for savings.", "Start by tracking your expenses and income, then allocate money accordingly.", "Budgeting is about balance—save, spend wisely, and invest in your future!"]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score?", "How do I check my credit score?", "How can I improve my credit score?"],
        "responses": ["A credit score is a number that reflects your financial reliability. The higher, the better!", "You can check your credit score using services like Credit Karma or your bank’s website.", "To improve your credit score, pay bills on time, reduce debt, and avoid unnecessary credit inquiries."]
    }
]

