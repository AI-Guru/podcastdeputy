import os
import gradio as gr
from fastapi import FastAPI
from enum import Enum
import numpy as np
import time
import logging
import uuid
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mpy
from moviepy.audio.fx.audio_fadein import audio_fadein
from moviepy.audio.fx.audio_fadeout import audio_fadeout
from moviepy.audio.AudioClip import AudioArrayClip
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
#default_aiupdate_podcast_title = open("assets/default_aiupdate_podcast_title.txt", "r").read()
#default_aiupdate_podcast_description = open("assets/default_aiupdate_podcast_description.txt", "r").read()
#default_aiupdate_podcast_text = open("assets/default_aiupdate_podcast_text.txt", "r").read()
#default_aiupdate_podcast_speech_path = "assets/default_aiupdate_podcast_speech.wav"
#default_aiupdate_podcast_video_path = "assets/default_aiupdate_podcast_video.mp4"
default_aiupdate_podcast_title = None
default_aiupdate_podcast_description = None
default_aiupdate_podcast_text = None
default_aiupdate_podcast_speech_path = None
default_aiupdate_podcast_video_path = None

default_sources_text = open("assets/defaultsources.txt", "r").read()
#default_sources_text = ""

# Load the environment variables.
dotenv.load_dotenv()

# Check if the environment variables are set.
if not os.getenv("ELEVENLABS_API_KEY"):
    raise ValueError("ELEVENLABS_API_KEY environment variable is not set. Get one from https://elevenlabs.io/app/settings/api-keys and set it in the .env file. See the .env.example file for an example.")
if not os.getenv("ELEVENLABS_VOICE_ID"):
    raise ValueError("ELEVENLABS_VOICE_ID environment variable is not set. Get one from https://elevenlabs.io/app/voice-lab and set it in the .env file. See the .env.example file for an example.")
if not os.getenv("ELEVENLABS_VOICE_2_ID"):
    raise ValueError("ELEVENLABS_VOICE_2_ID environment variable is not set. Get one from https://elevenlabs.io/app/voice-lab and set it in the .env file. See the .env.example file for an example.")
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set. Get one from https://console.anthropic.com/settings/keys and set it in the .env file. See the .env.example file for an example.")

text_to_speech_voice_1 = ElevenLabsTextToSpeech(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice_id=os.getenv("ELEVENLABS_VOICE_ID")
)
text_to_speech_voice_2 = ElevenLabsTextToSpeech(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice_id=os.getenv("ELEVENLABS_VOICE_2_ID")
)

class Utterance(BaseModel):
    person: str = Field(description="The person who is speaking.")
    text: str = Field(description="The text the person is speaking.")

class Dialogue(BaseModel):
    utterances: List[Utterance] = Field(description="The utterances in the dialogue.")


class Application:

    def __init__(self, development=False):
        self.development = development
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("Starting Gradio Application")

        self.chat_messages = []
        self.podcast_text = ""
        self.podcast_title = ""
        self.podcast_description = ""

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

            gr.Markdown("## AI Update")

            # Create two columns. The first has a chatbot plus text field plus button. The second has a text field and at the bottom a row with two buttons.
            with gr.Row():

                with gr.Column():
                    # Create the chatbot.
                    self.chatbot = ChatOllama()

                    # Create the chatbot's text field.
                    self.chatbot_element = gr.Chatbot(label="Chatbot", value=self.chat_messages, type="messages")

                    # Create the sources text field.
                    self.sources_element = gr.Textbox(lines=5, label="Sources", placeholder="Type here", value=default_sources_text)

                    self.instructions_element = gr.Textbox(lines=1, label="Instructions", placeholder="Type here", value="")

                    # Create the button to send the user's message.
                    self.send_button = gr.Button("Send")
            
                with gr.Column():

                    # Create the text fields.
                    self.podcast_title_field = gr.Textbox(lines=1, label="Title", placeholder="Type here", type="text", value=default_aiupdate_podcast_title)
                    self.podcast_description_field = gr.Textbox(lines=5, label="Description", placeholder="Type here", type="text", value=default_aiupdate_podcast_description)
                    self.podcast_text_field = gr.Textbox(lines=20, max_lines=20, label="Podcast Text", placeholder="Type here", type="text", value=default_aiupdate_podcast_text)

                    with gr.Row():
                        # Create the button to start the conversation.
                        self.speech_button = gr.Button("TTS")

                        # Create the button to reset the conversation.
                        self.video_button = gr.Button("Video")

                    with gr.Row():

                        # Create and audio player.
                        self.audio_player = gr.Audio(value=default_aiupdate_podcast_speech_path)

                        # Create video player.
                        self.video_player = gr.Video(value=default_aiupdate_podcast_video_path)

            # Create the event handlers.
            self.send_button.click(
                self.process_sources,
                inputs=[self.sources_element, self.instructions_element],
                outputs=[self.chatbot_element, self.podcast_title_field, self.podcast_description_field, self.podcast_text_field]
            )
            self.sources_element.submit(
                self.process_sources,
                inputs=[self.sources_element, self.instructions_element],
                outputs=[self.chatbot_element, self.podcast_title_field, self.podcast_description_field, self.podcast_text_field]
            )
            self.speech_button.click(
                self.text_to_speech,
                inputs=[self.podcast_text_field],
                outputs=[self.audio_player]
            )
            self.video_button.click(
                self.speech_to_video,
                inputs=[self.podcast_title_field, self.podcast_description_field, self.podcast_text_field, self.audio_player],
                outputs=[self.video_player]
            )

    def process_sources(self, sources, instructions):

        def compile_yield():
            return self.chat_messages, self.podcast_title, self.podcast_description, self.podcast_text

        # Split and strip the sources.
        urls = sources.split("\n")
        urls = [url.strip() for url in urls if url.strip() != ""]
        
        # See if all is okay.
        for url in urls:
            if not url.startswith("http"):
                self.add_chat_message("assistant", f"Invalid URL: {url}")
                yield compile_yield()
                raise StopIteration()
        
        # The system message.
        system_message = SystemMessagePromptTemplate.from_template_file("prompttemplates/aiupdate_system.txt", input_variables=[])
        system_message = system_message.format(
            name="Dr. Tristan",
        )

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
            human_message = HumanMessagePromptTemplate.from_template_file("prompttemplates/summarizetask.txt", input_variables=[])
            human_message = human_message.format(
                content=page_content,
            )
            processed_content = self.invoke_model(system_message, human_message)
            #processed_content = "Deutschland is in Sachen KI abgehängt. Das ist das Ergebnis einer Studie des Weltwirtschaftsforums. Deutschland belegt in einem Ranking von 105 Ländern den 17. Platz. Die Studie untersucht, wie gut Länder auf die KI-Revolution vorbereitet sind. Deutschland hat in den letzten Jahren an Boden verloren. In der Studie werden verschiedene Faktoren untersucht. Dazu gehören die Qualität der Forschung, die Verfügbarkeit von Daten und die politische Unterstützung. Deutschland hat in allen Bereichen schlecht abgeschnitten. Die USA und China sind die führenden Länder in Sachen KI. Sie haben die besten Forschungseinrichtungen und die meisten Daten. Die Politik unterstützt die KI-Entwicklung. Deutschland muss mehr in KI investieren, um den Anschluss nicht zu verlieren."

            # Handle errors.
            if "COULD NOT READ" in processed_content:
                self.add_chat_message("assistant", f"Could not process content: {url}. {processed_content}")
                yield compile_yield()
                continue

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

        # Get the current date. The day is a number. The month is a word. The year is a number.
        current_date_day = time.strftime("%d")
        current_date_month = time.strftime("%B")
        current_date_year = time.strftime("%Y")
        current_date = f"{current_date_day} {current_date_month} {current_date_year}"

        # Write the podcast text.
        parser = PydanticOutputParser(pydantic_object=Dialogue)
        parser_format_instructions = parser.get_format_instructions()
        human_message = HumanMessagePromptTemplate.from_template_file("prompttemplates/dialogue_text.txt", input_variables=[])
        human_message = human_message.format(
            contents=contents,
            current_date=current_date,
            instructions=instructions,
        )
        human_message.content += "\n\n" + parser_format_instructions
        dialogue = self.invoke_model(system_message, human_message)
        dialogue = parser.invoke(dialogue)
        #self.podcast_text_pydantic = dialogue
        self.podcast_text = dialogue.model_dump_json(indent=4)
        yield compile_yield()

        # Write the podcast title.
        human_message = HumanMessagePromptTemplate.from_template_file("prompttemplates/aiupdate_podcast_title.txt", input_variables=[])
        human_message = human_message.format(
            podcast_text=self.podcast_text,
        )
        self.podcast_title = self.invoke_model(system_message, human_message)
        self.podcast_title = time.strftime("%d.%m.%Y") + " - " + self.podcast_title.split("Dr. Tristans AI Update:")[1].strip()
        self.podcast_title = self.podcast_title.replace("Tristan", "TR15TAN")
        yield compile_yield()

        # Write the podcast description.
        human_message = HumanMessagePromptTemplate.from_template_file("prompttemplates/aiupdate_podcast_description.txt", input_variables=[])
        human_message = human_message.format(
            podcast_text=self.podcast_text,
        )
        self.podcast_description = self.invoke_model(system_message, human_message)
        self.podcast_description = self.podcast_description.replace("Tristan", "TR15TAN")
        for processed_source in self.processed_sources:
            self.podcast_description += f"\n\nQuelle: {processed_source['url']}"
        self.podcast_description += "\n\n"
        self.podcast_description += "Achtung: Dr. TR15TANs AI Update ist ein künstlich generierter Podcast. Die Informationen sind möglicherweise nicht korrekt."
        yield compile_yield()

        # Done.
        self.add_chat_message("assistant", f"Done processing sources.")
        yield compile_yield()

    def load_content(self, url, mode: str = "playwrightcustom", extract_body: bool = True):

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
        
        # Extract the body.
        if extract_body:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page_content, "html.parser")
            body = soup.find("body")
            if body is not None:
                page_content = body.get_text()

        # Done.
        return page_content


    def text_to_speech(self, text):

        # Parse the dialogue.
        dialogue = Dialogue.model_validate_json(text)
       
        # Create the output folder.
        output_folder = os.path.join("output", "dialogues")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Create the audio files.
        output_files = []
        for utterance_index, utterance in enumerate(dialogue.utterances):
            print(f"{utterance.person}: {utterance.text}")

            output_file = os.path.join(output_folder, f"{utterance_index:04d}-{utterance.person}.wav")
            if "Tristan" in utterance.person:
                text_to_speech = text_to_speech_voice_1
            elif "Robert" in utterance.person:
                text_to_speech = text_to_speech_voice_2
            else:
                raise ValueError(f"Unknown person: {utterance.person}")
            audio = text_to_speech.to_speech(utterance.text)
            with open(output_file, "wb") as f:
                f.write(audio)
            output_files.append(output_file)

        # Merge the audio files.
        audio_clips = [mpy.AudioFileClip(output_file) for output_file in output_files]
        audio_clip = mpy.concatenate_audioclips(audio_clips)
        output_file = os.path.join(output_folder, "dialogue.wav")
        audio_clip.write_audiofile(output_file)

        # Save the audio to a file.
        #if not os.path.exists("output"):
        #    os.makedirs("output")
        #filename = f"output/{uuid.uuid4()}.wav"
        #with open(filename, "wb") as f:
        #    f.write(audio)
        return output_file

    
    def speech_to_video(self, podcast_title, podcast_description, podcast_text, podcast_speech):

        print(f"Podcast title: {podcast_title}")
        print(f"Podcast description: {podcast_description}")
        print(f"Podcast text: {podcast_text[:100]}...")
        print(f"Podcast speech path: {podcast_speech}")
        assert isinstance(podcast_speech, tuple), f"Expected a tuple for the podcast title, but got {type(podcast_title)}"
        assert len(podcast_speech) == 2, f"Expected a tuple of length 2 for the podcast speech, but got {len(podcast_speech)}"
        assert isinstance(podcast_speech[0], int), f"Expected an integer for the podcast speech samplerate, but got {type(podcast_speech[0])}"
        assert isinstance(podcast_speech[1], np.ndarray), f"Expected a numpy array for the podcast speech data, but got {type(podcast_speech[1])}"
        podcast_speech_samplerate = podcast_speech[0]
        podcast_speech_data = podcast_speech[1]
        print(f"Podcast speech samplerate: {podcast_speech_samplerate}")
        print(f"Podcast speech data length: {len(podcast_speech_data)}")

        # Get the timestam. Format is DD.MM.YYYY.
        current_date = time.strftime("%d.%m.%Y")

        # Do some text transformations.
        podcast_title_short = podcast_title
        #podcast_title_short = time.strftime("%d.%m.%Y") + " - " + podcast_title.split("Dr. Tristans AI Update:")[1].strip()
        #podcast_title = podcast_title.replace("Tristan", "TR15TAN")
        #podcast_description = podcast_description.replace("Tristan", "TR15TAN")
        print(f"Podcast title: {podcast_title}")
        print(f"Podcast title short: {podcast_title_short}")
        print(f"Podcast description: {podcast_description}")
        print(f"Podcast text: {podcast_text[:100]}...")

        # Make sure the media files exist.
        wallpaper_path = "assets/aiupdate_wallpaper.jpg"
        assert os.path.exists(wallpaper_path), f"Wallpaper path does not exist: {wallpaper_path}"
        logo_path = "assets/aiupdate_logo.jpg"
        assert os.path.exists(logo_path), f"Logo path does not exist: {logo_path}"
        theme_music_path = "assets/aiupdate_theme.mp3"
        assert os.path.exists(theme_music_path), f"Theme music path does not exist: {theme_music_path}"

        # Render the logo and the title onto the wallpaper.
        wallpaper = Image.open(wallpaper_path)
        logo_scale = 0.5
        logo = Image.open(logo_path)
        logo = logo.resize((int(logo.width * logo_scale), int(logo.height * logo_scale)))
        wallpaper.paste(logo, (50, 175))
        draw = ImageDraw.Draw(wallpaper)

        # Find the font garamond.
        import matplotlib.font_manager as fm
        font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        found = False
        for font_path in font_paths:
            if "garamond" in font_path.lower():
                found = True
                break
        assert found, f"Could not find Garamond font"

        # Use a default font.
        font_size = 40
        font = ImageFont.truetype(font_path, font_size)

        # Calculate the bounding box of the text
        text_bbox = draw.textbbox((50, 600), podcast_title_short, font=font)

        # Make the text smaller if it is too big.
        while text_bbox[2] > wallpaper.width - 50:
            font_size -= 5
            font = ImageFont.truetype(font_path, font_size)
            text_bbox = draw.textbbox((50, 600), podcast_title_short, font=font)

        # Make the text bounding box a bit bigger.
        offset = 10
        text_bbox = (text_bbox[0], text_bbox[1], text_bbox[2] + 2 *offset, text_bbox[3] + offset)

        # Draw a white rectangle around the text
        draw.rectangle(text_bbox, outline=(255, 255, 255), fill=(255, 255, 255), width=5)

        # Draw the text
        draw.text((50 + offset, 600 + offset), podcast_title_short, fill=(0, 0, 0), font=font)

        # Get the current date. Format is YYYYMMDD.
        current_date = time.strftime("%Y%m%d")

        # Save the wallpaper
        wallpaper.save(f"output/{current_date}_wallpaper.jpg")

        # Load the theme music. Use the first n seconds. Add a fade-in and fade-out.
        clip_length_seconds = 16
        fade_in_seconds = 2
        fade_out_seconds = 4
        theme_music = mpy.AudioFileClip(theme_music_path)
        theme_music = theme_music.subclip(0, clip_length_seconds)
        theme_music = audio_fadein(theme_music, fade_in_seconds) # Apply fade-in
        theme_music = audio_fadeout(theme_music, fade_out_seconds) # Apply fade-out
        theme_music = theme_music.set_duration(clip_length_seconds)

        # First save the audio data to a temporary file
        import soundfile as sf
        temp_speech_path = f"output/{current_date}_temp_speech.wav"
        sf.write(temp_speech_path, podcast_speech_data, podcast_speech_samplerate)

        # Then load it using AudioFileClip
        podcast_speech = mpy.AudioFileClip(temp_speech_path)

        # Rest of your code remains the same
        audio_track = mpy.concatenate_audioclips([theme_music, podcast_speech])
        audio_track_path = f"output/{current_date}_audio.mp3"
        audio_track.write_audiofile(audio_track_path)

        # Add the wallpaper as the background
        video = mpy.ImageClip(f"output/{current_date}_wallpaper.jpg")
        video = video.set_duration(audio_track.duration)
        video = video.set_audio(audio_track)

        # Save the video
        video_path = f"output/{current_date}_video.mp4"
        video.write_videofile(video_path, codec="libx264", audio_codec="aac", fps=24)

        # Optional: Clean up the temporary file
        os.remove(temp_speech_path)

        return video_path

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

        return self.invoke_model_messages([system_message, human_message])


    def invoke_model_messages(self, messages):
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