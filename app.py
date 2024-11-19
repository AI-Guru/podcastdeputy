import os
import gradio as gr
from fastapi import FastAPI
from enum import Enum
import time
import logging
import uuid
import dotenv
import langchain_core
from langchain_core.prompts.chat import HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader, PlaywrightURLLoader
from source.texttospeech import ElevenLabsTextToSpeech


# Configuration.
default_canvas_text = open("assets/defaulttext.txt", "r").read()
default_canvas_text = "Herzlich willkommen zum AI Update."

default_sources_text = open("assets/defaultsources.txt", "r").read()

# Load the environment variables.
dotenv.load_dotenv()

text_to_speech = ElevenLabsTextToSpeech(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice_id="9gSkuKCHRczfU5aLq1qU"
    #voice_id="yCJwUkOEJeSHB8xTh6HQ", #Robert?
    #voice_clone_name="Robert",
    #voice_clone_description="Ein 46-jähriger deutscher Mann!",
    #voice_clone_samples=["assets/voice_sample_robert.mp3"]
)

class Application:

    def __init__(self, development=False):
        self.development = development
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("Starting Gradio Application")

        self.chat_messages = []
        self.podcast_text = ""

        self.processed_sources = []

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
                    self.chatbot_element = gr.Chatbot(label="Chatbot", value=self.chat_messages, type="messages")

                    # Create the user's text field.
                    self.sources_element = gr.Textbox(lines=5, label="Sources", placeholder="Type here", value=default_sources_text)

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
                self.process_sources,
                inputs=[self.sources_element],
                outputs=[self.chatbot_element, self.canvas_field]
            )
            self.sources_element.submit(
                self.process_sources,
                inputs=[self.sources_element],
                outputs=[self.chatbot_element, self.canvas_field]
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

    def process_sources(self, sources):

        def compile_yield():
            return self.chat_messages, self.podcast_text

        urls = sources.split("\n")
        urls = [url.strip() for url in urls if url.strip() != ""]
        
        # See if all is okay.
        for url in urls:
            if not url.startswith("http"):
                self.add_chat_message("assistant", f"Invalid URL: {url}")
                yield compile_yield()
                raise StopIteration()
        
        # Process the sources.
        for url in urls:

            # If it already has been processed, skip.
            if any([processed_source["url"] == url for processed_source in self.processed_sources]):
                self.add_chat_message("assistant", f"URL already processed: {url}")
                yield compile_yield()
                continue

            self.add_chat_message("assistant", f"Processing URL: {url}")
            yield compile_yield()

            # Skip if the URL is already processed.
            if any([processed_source["url"] == url for processed_source in self.processed_sources]):
                self.add_chat_message("assistant", f"URL already processed: {url}")
                yield compile_yield()
                continue

            # Load the document and get the page content.
            page_content = self.load_content(url)

            # Process the content.
            system_message = SystemMessagePromptTemplate.from_template_file("prompttemplates/system.txt", input_variables=[])
            system_message = system_message.format(
                name="Maximilian Durchfahrtshöhe",
            )
            human_message = HumanMessagePromptTemplate.from_template_file("prompttemplates/summarizetask.txt", input_variables=[])
            human_message = human_message.format(
                content=page_content,
            )
            processed_content = self.invoke_model(system_message, human_message)

            # Add the processed content to the chat.
            self.add_chat_message("assistant", f"Processed content: {processed_content}")
            yield compile_yield()

            # Save the processed source.
            processed_source = {
                "url": url,
                "original_text": page_content,
                "processed_text": processed_content,
            }
            self.processed_sources.append(processed_source)

            # Count the number of words.
            words = page_content.split(" ")
            self.add_chat_message("assistant", f"Number of words roughly: {len(words)}")

            yield compile_yield()


        # Update the chat.
        self.add_chat_message("assistant", f"Done processing sources. Now processing the sources.")
        yield compile_yield()

        # Now process the sources.
        contents = "Here are the processed sources:\n\n"
        for processed_source_index, processed_source in enumerate(self.processed_sources):
            contents += f"Source {processed_source_index + 1}:\n"
            contents += "```\n"
            contents += processed_source["processed_text"]
            contents += "\n```\n\n" 

        # Get the current date. Format is DD.MM.YYYY.
        current_date = time.strftime("%d.%m.%Y")

        # Language is German.
        language = "German"

        # Process the document.
        human_message = HumanMessagePromptTemplate.from_template_file("prompttemplates/rewritetask.txt", input_variables=[])
        human_message = human_message.format(
            contents=contents,
            current_date=current_date,
            language=language
        )
        self.podcast_text = self.invoke_model(system_message, human_message)

        self.add_chat_message("assistant", f"Done processing sources.")
        yield compile_yield()

    def load_content(self, url, mode: str = "playwrightcustom"):

        if mode == "webbase":
            loader = WebBaseLoader(url)
            documents = loader.load()
            assert len(documents) == 1, f"Expected 1 document, but got {len(documents)}"
            document = documents[0]
            page_content = document.page_content

        elif mode == "playwright":
            loader = PlaywrightURLLoader(url)
            documents = loader.load()
            assert len(documents) == 1, f"Expected 1 document, but got {len(documents)}"
            document = documents[0]
            page_content = document.page_content

        elif mode == "playwrightcustom":
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url)
                content = page.content()
                browser.close()
            page_content = content
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return page_content


    def text_to_speech(self, text):
        
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
    
    def sources_to_text(self):

        text = ""

        for processed_source in self.processed_sources:
            url = processed_source["url"]
            original_text = processed_source["original_text"]
            processed_text = processed_source["processed_text"]

            text += processed_text + "\n\n"

        return text


    def add_chat_message(self, role, content, is_article_message=False):
        assert role in ["user", "assistant"], f"Invalid role: {role}"
        new_message = {"role": role, "content": content}
        self.chat_messages.append(new_message)
        if is_article_message:
            self.article_messages.append(new_message)


    def invoke_model(self, system_message, human_message):
        assert isinstance(system_message, langchain_core.messages.system.SystemMessage), f"Expected a string for the system message, but got {type(system_message)}"
        assert isinstance(human_message, langchain_core.messages.human.HumanMessage), f"Expected a string for the human message, but got {type(human_message)}"
        self.logger.info(f"System message: {system_message}")
        self.logger.info(f"Human message: {human_message}")

        return self.invoke_model_messags([system_message, human_message])


    def invoke_model_messags(self, messages):
        assert isinstance(messages, list), f"Expected a list of messages, but got {type(messages)}"
        llm = self.get_llm()
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        text = chain.invoke({})
        self.logger.info(f"Model response: {text}")
        return text
    

    def get_llm(self):
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.7,
            max_tokens=8192,
        )
    





# FastAPI and Gradio integration
fast_api_app = FastAPI()

# Initialize Gradio
gradio_app = Application(development=True)  # Create an instance of the GradioApp class
gradio_app.build_interface()  # Build the interface

# Mount Gradio app onto FastAPI
app = gr.mount_gradio_app(fast_api_app, gradio_app.demo, path="/")