import elevenlabs as ELabs


def playBytesAudio(audio:bytes):
    ELabs.play(audio)

class Text2Speech():
    def __init__(self) -> None:
        ELabs.set_api_key("38429343292a32ff2248a9b268658a87")
        pass

    def setApiKey(self,key:str):
        ELabs.set_api_key(key)

    def GenerateAudio(self,text:str) -> bytes:
        return ELabs.generate(
          text=text,
            voice=ELabs.Voice(
                voice_id="Bella",
                settings=ELabs.VoiceSettings(stability=0.61, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
            ),
          model="eleven_multilingual_v2"
        )

    def GenerateAudioStream(self,text:str) -> bytes:
        return ELabs.generate(
          text=text,
          model="eleven_multilingual_v2",
          stream=True
        )

    def playAudio(self,audio:bytes):
        ELabs.play(audio)
    
    def streamAudio(self, audio):
        ELabs.stream(audio)


if __name__ == "__main__":
    t2s = Text2Speech()

    a = t2s.GenerateAudioStream("Test2")
    t2s.streamAudio(a)
        