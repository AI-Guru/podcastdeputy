import os
from typing import Iterator
import numpy as np
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

class ElevenLabsTextToSpeech:

    def __init__(
            self, 
            api_key:str,
            voice_id:str = None,
            voice_clone_name:str = None,
            voice_clone_description:str = None,
            voice_clone_samples:list[str] = None, 
            model_id:str = "eleven_multilingual_v2",
            langauge_code:str = "de_DE"
        ):

        has_voice_id = voice_id is not None
        has_voice_clone = any([voice_clone_name is not None, voice_clone_description is not None, voice_clone_samples is not None])
        if not has_voice_id and not has_voice_clone:
            raise ValueError("Either voice_id or voice_clone_name must be provided.")
        if has_voice_id and has_voice_clone:
            raise ValueError("Either voice_id or voice_clone_name must be provided, not both.")
        if has_voice_clone:
            if not all([voice_clone_name, voice_clone_description, voice_clone_samples]):
                raise ValueError("voice_clone_name, voice_clone_description, and voice_clone_samples must be provided.")

        self.api_key = api_key
        self.model_id = model_id
        self.language_code = langauge_code

        # Create the client.
        self.client = ElevenLabs(
            api_key=self.api_key
        )

        # Handle the voices.
        if voice_id is not None:
            self.voice=Voice(
                voice_id=voice_id,
                settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
            )
        elif voice_clone_samples is not None and voice_clone_name is not None:
            self.voice = self.client.clone(
                name=voice_clone_name,
                description=voice_clone_description,
                files=voice_clone_samples,
            )


    def to_speech_requests(self, text):
        import requests

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"

        payload = {
            "text": text,
            "model_id": self.model_id,
            #"language_code": self.language_code,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                #"style": 123,
                #"use_speaker_boost": True
            },
            #"pronunciation_dictionary_locators": [
            #    {
            #        "pronunciation_dictionary_id": "<string>",
            #        "version_id": "<string>"
            #    }
            #],
            #"seed": 123,
            #"previous_text": "<string>",
            #"next_text": "<string>",
            #"previous_request_ids": ["<string>"],
            #"next_request_ids": ["<string>"],
            #"use_pvc_as_ivc": True,
            #"apply_text_normalization": "auto"
        }
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

        response = requests.request("POST", url, json=payload, headers=headers)

        # Handle the response.
        if response.status_code != 200:
            raise ValueError(f"Failed to generate audio. {response.text}")
        #


        print(response.text)

        return response.content


    def to_speech(self, text):


        audio = self.client.generate(
            text=text, 
            voice=self.voice,
            model=self.model_id
        )
        if isinstance(audio, Iterator):
            audio = b"".join(audio)
        else:
            raise ValueError("The audio is not an iterator.")
        
        return audio
