import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
import speech_recognition as sr
import pyttsx3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
from langchain.memory import ChatMessageHistory
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from emotions_detetion import prediction

from modules.layout import Layout
from modules.utils import Utilities
from gtts import gTTS

st.set_page_config(
    layout="wide",
    page_icon="💬",
    page_title="Modus ETP Inc AI assistant | Chat-Bot 🤖",
)

if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant",
                        "content": "Welcome to our voice ai assistant"}]

layout, utils = Layout(), Utilities()

layout.show_header()

user_api_key = utils.load_api_key()
os.environ["GOOGLE_API_KEY"] = user_api_key


prompt = """You are a chatbot having a conversation with a human your name is Modues ETP , subject should be about Indian states, culture , etc note your answers should be correct and short to the point , also you should consider the emotions of the user.
here is the user question now : {question}
here is the user previous questions use it and link the chating between you and the user laest answer.
history:[{history}]
and there is user emotions use it to understand it's mood : {user_emotions}
"""

prompt_emotions = """
You are a EXPRESSIONS detection bot your task will be given an user input {user_input} and you should classify it's expression to one of these class : ["Awkwardness","Fear","Confusion","Doubt","Anxiety","Anger","Determination","Interest","Distress","Realization","Pride","Concentration","Amusement","Sadness","Sympathy","Calmness","Boredom","Joy","Surprise(Positive)","Surprise (negative)","Admiration","Desire","Disappointment","Contempt","Contemplation"] 
                Output : your output should be the class name with the score given that you should given all these classes ration so that they sum to 1
                you should output the height three classes with it's scores in the formate =>  class_name : score eash (with only two numbers)
                and note each class in a seperate line 
                after each class add \n\n 
"""
if not user_api_key:
    layout.show_api_key_missing()
else:     
    prompt = PromptTemplate(input_variables=["question","history","user_emotions"], template = prompt)

    prompt_emotions = PromptTemplate(input_variables=["user_input"], template = prompt_emotions)
    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                 convert_system_message_to_human=True,
                                 temperature=0)
    chat_llm_chain = LLMChain(llm=llm,
                              prompt=prompt,
                              verbose=True,)   
    chat_llm_chain_emotions = LLMChain( 
        llm=llm,
        prompt=prompt_emotions,
        verbose=True,
    )

    recognizer = sr.Recognizer()
    # Add a selectbox to the sidebar
    option = st.sidebar.selectbox(
        'Choose a model',
        ('Audio Classification Model', 'Text Classification Model'),
        index=1  # Setting the default index to 1, which corresponds to 'LLM Model'
    )

    def main():
        audio_bytes = audio_recorder()
        if audio_bytes:
            audio_file_path = "temp_file.wav"
            with open(audio_file_path, "wb") as f:
                f.write(audio_bytes)
            # st.audio(audio_bytes, format="audio/wav")

            # Load the saved audio file
            print("loading audio file")
            with sr.AudioFile(audio_file_path) as source:
                # Record the audio file into an AudioData instance
                audio_data = recognizer.record(source)
                try:
                    user_text = recognizer.recognize_google(audio_data)
                    if option =="Text Classification Model":
                        human_classes = chat_llm_chain_emotions.invoke({"user_input": user_text})
                        human_classes = human_classes["text"]
                    else:
                        human_classes = prediction(audio_file_path) 
                        human_classes = human_classes[0]     
                    st.markdown(
                    f"""
                    <div style='text-align: center;'>
                        <h5>👩: {user_text}</h5>
                    </div>
                    """,
                    unsafe_allow_html=True,
                    )
                    lines = human_classes.strip().split('\n')
                    st.markdown(
                    """
                    <div style='text-align: center;'>
                        <h5>✨ Detected EXPRESSIONS:</h5>
                    </div>
                    """,
                        unsafe_allow_html=True,
                      )
                    # Render each line in Markdown with HTML div
                    for line in lines:
                        st.markdown(
                            f"""
                            <div style='text-align: center;'>
                                <h5>{line}</h5>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.session_state.messages.append({"role": "user", "content": f"{user_text}  \n \n ✨ user mode : {human_classes}"})
                    recent_msgs = st.session_state.messages[-10:]
                    # text = chat_llm_chain.predict(question=text,history=recent_msgs)
                    response = chat_llm_chain.invoke({'question':user_text , "history":recent_msgs,"user_emotions":human_classes})
                    print("response",response)  
                    text = response["text"]  
                    if option =="Text Classification Model":
                        bot_classes = chat_llm_chain_emotions.invoke({"user_input": text})
                        bot_classes = bot_classes["text"]                
                             
                    else:
                        myobj = gTTS(text=text, lang="en", slow=False)
                        audio_file_path = "model.wav"
                        myobj.save(audio_file_path)
                        bot_classes = prediction(audio_file_path)  
                        bot_classes = bot_classes[0] 
                        
                    st.session_state.messages.append({"role": "assistant", "content": f"{text}  \n \n ✨ Bot mode : {bot_classes}"})
                    engine = pyttsx3.init() # object creation
                    engine.setProperty('rate', 150)     # setting up new voice rate
                    engine.setProperty('volume',1)    # setting up volume level  between 0 and 1
                    voices = engine.getProperty('voices')       #getting details of current voice
                    #engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
                    engine.setProperty('voice', voices[2].id)   #changing index, changes voices. 1 for female
                    st.markdown(
                    f"""
                    <div style='text-align: center;'>
                        <h5>🤖: {text}</h5>
                    </div>
                    """,
                    unsafe_allow_html=True,
                    )
                    lines = bot_classes.strip().split('\n')

                    # Display each line separately
                    st.markdown(
                    """
                    <div style='text-align: center;'>
                        <h5>✨ Detected EXPRESSIONS:</h5>
                    </div>
                    """,
                        unsafe_allow_html=True,
                      )

                    # Render each line in Markdown with HTML div
                    for line in lines:
                        st.markdown(
                            f"""
                            <div style='text-align: center;'>
                                <h5>{line}</h5>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        
                    engine.say(text)
                    engine.runAndWait()
                    # engine.stop()
                    engine = None
                except sr.UnknownValueError:
                    print("Google Web Speech API could not understand the audio")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Web Speech API; {e}")
        for idx, msg in enumerate(st.session_state.messages):
            print(len(st.session_state.messages))
            if len(st.session_state.messages)!=0:
                st.sidebar.chat_message(msg["role"]).write(msg["content"])
    if __name__ == "__main__":
        main()