from typing import Iterator
import numpy as np
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

class ElevenLabsTextToSpeech:

    def __init__(
            self, 
            api_key,
            voice_id,
            model_id="eleven_multilingual_v2",
            langauge_code="de_DE"
        ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.language_code = langauge_code


    def to_speech_requests(self, text):
        import requests

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"

        payload = {
            "text": text,
            "model_id": self.model_id,
            #"language_code": self.language_code,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
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
        client = ElevenLabs(
            api_key=self.api_key
        )
        voice=Voice(
            voice_id=self.voice_id,
            settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
        )
        audio = client.generate(
            text=text, 
            voice=voice,
            model=self.model_id
        )
        if isinstance(audio, Iterator):
            audio = b"".join(audio)
        else:
            raise ValueError("The audio is not an iterator.")
        
        return audio
