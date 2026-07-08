import os
from dotenv import load_dotenv
from groq import Groq
import asyncio
import edge_tts
import pygame

import torch 
from silero_vad import load_silero_vad, VADIterator


import sounddevice as sd          # NEW: record from the mic
import soundfile as sf            # NEW: save audio to a .wav file
import numpy as np                # NEW: hold the audio samples

# --- Setup ---
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# --- Voice OUT (text -> speech) via Microsoft Edge TTS (natural English voices) ---
pygame.mixer.init()
VOICE = "en-US-ChristopherNeural"   # natural English MALE voice

def speak(text):
    asyncio.run(edge_tts.Communicate(text, VOICE).save("reply.mp3"))
    pygame.mixer.music.load("reply.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():     # wait until it finishes speaking
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()              # release the file for the next turn



# --- Voice IN, part 1 — record until you stop talking, using Silero VAD ---
SAMPLE_RATE = 16000   # Silero works at 16 kHz
CHUNK = 512           # Silero processes exactly 512 samples at a time (~32 ms)

vad_model = load_silero_vad()   # the neural VAD — runs locally, loaded once

def record_until_silence():
    print("🎤 Listening... (just start talking)")
    vad = VADIterator(vad_model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=800)
    frames = []
    speech_started = False
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while True:
            block, _ = stream.read(CHUNK)                   # 512 samples of audio
            frames.append(block.copy())
            chunk = torch.from_numpy(block[:, 0].copy())    # 1-D tensor for the model
            event = vad(chunk)                              # ask Silero: speech start/end?
            if event:
                if "start" in event:
                    speech_started = True
                if "end" in event and speech_started:
                    break                                   # Silero decided you're done
    vad.reset_states()                                      # reset for the next turn
    audio = np.concatenate(frames, axis=0)
    sf.write("input.wav", audio, SAMPLE_RATE)
    return "input.wav"


# --- NEW: Voice IN, part 2 — turn the recording into text (Groq Whisper) ---
def transcribe(wav_path):
    with open(wav_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(wav_path, f.read()),
            model="whisper-large-v3-turbo",
        )
    return result.text.strip()

# --- Load everything we know about you ---
with open("about_me.md", "r", encoding="utf-8") as f:
    knowledge = f.read()

# --- Persona + grounding (unchanged) ---
system_prompt = f"""You are the digital twin of Vilohit. You speak AS Vilohit, in the first person ("I built...", "I worked on...").

Answer using ONLY the information in the KNOWLEDGE section below.
If you're asked something the knowledge doesn't cover, say honestly: "I haven't shared anything about that yet." Do NOT invent facts, dates, numbers, jobs, or projects.

Keep replies friendly, natural, and fairly short — like you're chatting.

--- KNOWLEDGE ---
{knowledge}
--- END KNOWLEDGE ---
"""

# --- Memory ---
messages = [{"role": "system", "content": system_prompt}]

print("🗣️  Voice chat with Vilohit's digital twin — just talk!")
print("    Say 'goodbye' to end.\n")

while True:
    # 1. LISTEN — automatically detects when you stop talking
    wav = record_until_silence()
    user_input = transcribe(wav)
    print("You said:", user_input)
    if not user_input:
        print("(didn't catch that — try again)\n")
        continue

    # spoken exit
    if any(word in user_input.lower() for word in ["goodbye", "bye", "stop chatting"]):
        speak("Goodbye! It was great talking to you.")
        print("Bye! 👋")
        break

    # 2. THINK — same brain
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    # 3. SPEAK
    print("Vilohit:", reply, "\n")
    speak(reply)

