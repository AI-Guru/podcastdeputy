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
from langchain_core.pydantic_v1 import BaseModel, Field, validator
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


# Load environment variables
dotenv.load_dotenv()

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7,
    max_tokens=8192,
)

text_to_speech_robert = ElevenLabsTextToSpeech(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice_id="VJmQeeqhTBZ2B4l4yyq2",
    #voice_clone_name="Robert",
    #voice_clone_description="Ein 46-jähriger deutscher Mann!",
    #voice_clone_samples=["assets/voice_sample_robert.mp3"]
)

text_to_speech_tristan = ElevenLabsTextToSpeech(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice_id="VJmQeeqhTBZ2B4l4yyq2",
    #voice_clone_name="Tristan",
    #voice_clone_description="Ein 43-jähriger deutscher Mann.",
    #voice_clone_samples=["assets/voice_sample_tristan.mp3"]
)
text_to_speech_martin = ElevenLabsTextToSpeech(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice_id="de_DE_3",
    #voice_clone_name="Martin",
    #voice_clone_description="Ein 44-jähriger deutscher Mann!",
    #voice_clone_samples=["assets/voice_sample_martin.mp3"]
)


class Utterance(BaseModel):
    person: str = Field(description="The person who is speaking.")
    text: str = Field(description="The text the person is speaking.")

class Dialogue(BaseModel):
    utterances: List[Utterance] = Field(description="The utterances in the dialogue.")

parser = PydanticOutputParser(pydantic_object=Dialogue)
parser_format_instructions = parser.get_format_instructions()



system_message = SystemMessagePromptTemplate.from_template_file("prompttemplates/dialogue_system.txt", input_variables=[])
system_message = system_message.format()
system_message = system_message.content
human_message = HumanMessagePromptTemplate.from_template_file("prompttemplates/dialogue_task.txt", input_variables=[])
human_message = human_message.format()
human_message = human_message.content
human_message += "\n\n" + parser_format_instructions

print(system_message)
print(human_message)

# Invoke the model.
response = llm.invoke(
    [("system", system_message), ("human", human_message)]
)
dialogue = parser.invoke(response)

# Create the output folder.
output_folder = "output/dialogues"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create the audio files.
output_files = []
for utterance_index, utterance in enumerate(dialogue.utterances):
    print(f"{utterance.person}: {utterance.text}")

    output_file = os.path.join(output_folder, f"{utterance_index:04d}-{utterance.person}.wav")
    if utterance.person == "Tristan":
        text_to_speech = text_to_speech_tristan
    elif utterance.person == "Robert":
        text_to_speech = text_to_speech_robert
    elif utterance.person == "Martin":
        text_to_speech = text_to_speech_martin
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

