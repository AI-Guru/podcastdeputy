import os
import gradio as gr
from fastapi import FastAPI
from enum import Enum
import time
import logging
import uuid
import dotenv
from langchain_core.prompts.chat import HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from source.texttospeech import ElevenLabsTextToSpeech


# Configuration.
default_canvas_text = open("assets/defaulttext.txt", "r").read()
default_canvas_text = "Herzlich willkommen zum AI Update."

# Load the environment variables.
dotenv.load_dotenv()


class Application:

    def __init__(self, development=False):
        self.development = development
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("Starting Gradio Application")

        #self.text_to_speech = ElevenLabsTextToSpeech(
        #    api_key=os.getenv("ELEVENLABS_API_KEY"),
       #     voice="brian"
        #)

    def build_interface(self):

        # CSS to deacivate the footer.
        css = "footer {visibility: hidden}"

        # Create the main window.
        with gr.Blocks(theme=gr.themes.Soft(), css=css) as self.demo:

            # Create two columns. The first has a chatbot plus text field plus button. The second has a text field and at the bottom a row with two buttons.
            with gr.Row():

                with gr.Column():
                    # Create the chatbot.
                    self.chatbot = ChatOllama()

                    # Create the chatbot's text field.
                    self.chatbot_text_field = gr.Textbox(lines=5, label="Chatbot", placeholder="Chatbot's response")

                    # Create the user's text field.
                    self.user_text_field = gr.Textbox(lines=5, label="User", placeholder="Type here")

                    # Create the button to send the user's message.
                    self.send_button = gr.Button("Send")

            
                with gr.Column():

                    # Create the text field for the user's name.
                    self.canvas_field = gr.Textbox(lines=20, max_lines=20, value=default_canvas_text)

                    with gr.Row():
                        # Create the button to start the conversation.
                        self.speech_button = gr.Button("Text to Speech")

                        # Create the button to reset the conversation.
                        self.download_button = gr.Button("Download Text")

                    # Create and audio player.
                    self.audio_player = gr.Audio()

            # Create the event handlers.
            self.send_button.click(
                self.process_user_message,
                inputs=[self.user_text_field],
                outputs=[self.chatbot_text_field, self.canvas_field]
            )
            self.user_text_field.submit(
                self.process_user_message,
                inputs=[self.user_text_field],
                outputs=[self.chatbot_text_field, self.canvas_field]
            )
            self.speech_button.click(
                self.text_to_speech,
                inputs=[self.canvas_field],
                outputs=[self.audio_player]
            )
            self.download_button.click(
                self.download_text,
                inputs=[self.canvas_field]
            )

    def process_user_message(self, user_text):
        return
    
    def text_to_speech(self, text):
        text_to_speech = ElevenLabsTextToSpeech(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="9gSkuKCHRczfU5aLq1qU"
            #voice_id="yCJwUkOEJeSHB8xTh6HQ"
        )
        audio = text_to_speech.to_speech(text)

        # Save the audio to a file.
        if not os.path.exists("output"):
            os.makedirs("output")
        filename = f"output/{uuid.uuid4()}.wav"
        with open(filename, "wb") as f:
            f.write(audio)
        return filename

    
    def download_text(self, text):
        return
    





# FastAPI and Gradio integration
fast_api_app = FastAPI()

# Initialize Gradio
gradio_app = Application(development=True)  # Create an instance of the GradioApp class
gradio_app.build_interface()  # Build the interface

# Mount Gradio app onto FastAPI
app = gr.mount_gradio_app(fast_api_app, gradio_app.demo, path="/")