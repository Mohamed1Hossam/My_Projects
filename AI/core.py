import os
import pyttsx3
import speech_recognition as sr
from abc import ABC, abstractmethod
from typing import Optional


# ====== Text-to-Speech (TTS) ======
class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()

    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()


# ====== Speech Recognition (STT) ======
class Listener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def listen(self) -> Optional[str]:
        with self.microphone as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
        try:
            command = self.recognizer.recognize_google(audio)
            print(f"Heard: {command}")
            return command.lower()
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None


# ====== Command Interface (Liskov Compliant) ======
class Command(ABC):
    @abstractmethod
    def match(self, command_text: str) -> bool:
        pass

    @abstractmethod
    def execute(self, command_text: str):
        pass


# ====== Specific Commands ======
class OpenAppCommand(Command):
    def match(self, command_text: str) -> bool:
        return "open" in command_text

    def execute(self, command_text: str):
        if "chrome" in command_text:
            os.system("start chrome")
        elif "vscode" in command_text or "vs code" in command_text:
            os.system("code")
        else:
            print("App not recognized")


class ReadFileCommand(Command):
    def match(self, command_text: str) -> bool:
        return "read file" in command_text

    def execute(self, command_text: str):
        try:
            with open("example.txt", "r") as f:
                content = f.read()
                print(content)
        except FileNotFoundError:
            print("File not found")


# ====== Jarvis Assistant Core ======
class Jarvis:
    def __init__(self):
        self.speaker = Speaker()
        self.listener = Listener()
        self.commands: list[Command] = [OpenAppCommand(), ReadFileCommand()]

    def run(self):
        self.speaker.speak("Hello sir, I am online.")
        while True:
            command_text = self.listener.listen()
            if command_text:
                for command in self.commands:
                    if command.match(command_text):
                        command.execute(command_text)
                        break
                else:
                    self.speaker.speak("Command not recognized.")


if __name__ == "__main__":
    jarvis = Jarvis()
    jarvis.run()
