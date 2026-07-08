import os, re, sys, asyncio
from dotenv import load_dotenv
from groq import Groq
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator
import edge_tts
import pygame

# --- Setup ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))          # ears + brain
pygame.mixer.init()
vad_model = load_silero_vad()

SAMPLE_RATE = 16000
CHUNK = 512
VOICE = "en-US-GuyNeural"

# Windows + edge-tts: use the selector loop and ONE reusable loop (avoids "event loop closed")
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
tts_loop = asyncio.new_event_loop()

# --- Knowledge ---
with open("about_me.md", "r", encoding="utf-8") as f:
    knowledge = f.read()

# --- Persona + TIGHTENED grounding (fixes the invented-birthday leak) ---
system_prompt = f"""You are the digital twin of Vilohit. You speak AS Vilohit, in the first person ("I built...", "I worked on...").

Answer using ONLY the information in the KNOWLEDGE section below. Keep replies short — 1 to 3 sentences — because they are spoken aloud.

GROUNDING RULES (strict):
- Never invent or guess facts, numbers, dates, jobs, or projects.
- You do NOT know Vilohit's date of birth, age, salary, address, or relationship status — none of that is below. If asked, politely say you'd rather not share that, instead of guessing or calculating it.
- Never infer a personal fact from a hint (e.g. do not guess a birth year from an email address).
- If something isn't in the knowledge, say: "I haven't shared anything about that."

--- KNOWLEDGE ---
{knowledge}
--- END KNOWLEDGE ---
"""
messages = [{"role": "system", "content": system_prompt}]

# ============================================================
# 1. EARS — record until you stop talking (Silero VAD), then transcribe
# ============================================================
def listen():
    print("🎤 Listening...")
    vad = VADIterator(vad_model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=800)
    frames, started = [], False
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while True:
            block, _ = stream.read(CHUNK)
            frames.append(block.copy())
            event = vad(torch.from_numpy(block[:, 0].copy()))
            if event:
                if "start" in event: started = True
                if "end" in event and started: break
    vad.reset_states()
    sf.write("input.wav", np.concatenate(frames, axis=0), SAMPLE_RATE)
    with open("input.wav", "rb") as f:
        result = client.audio.transcriptions.create(
            file=("input.wav", f.read()), model="whisper-large-v3-turbo")
    return result.text.strip()

# ============================================================
# 2. BRAIN — stream the reply, yielding ONE SENTENCE at a time
# ============================================================
def stream_sentences(messages):
    buffer = ""
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages, stream=True)
    for part in stream:
        buffer += (part.choices[0].delta.content or "")
        pieces = re.split(r'(?<=[.!?])\s+', buffer)   # split only at sentence ends
        if len(pieces) > 1:
            for s in pieces[:-1]:
                if s.strip(): yield s.strip()
            buffer = pieces[-1]
    if buffer.strip(): yield buffer.strip()

# ============================================================
# 3. MOUTH — synthesize a sentence, play it, and watch for BARGE-IN
# ============================================================
def synth(sentence, path="reply.mp3"):
    tts_loop.run_until_complete(edge_tts.Communicate(sentence, VOICE).save(path))

def play_with_bargein(path, mic, vad):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        block, _ = mic.read(CHUNK)                       # listen WHILE speaking
        event = vad(torch.from_numpy(block[:, 0].copy()))
        if event and "start" in event:                   # you started talking!
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            return True                                  # interrupted
    pygame.mixer.music.unload()
    return False

# ============================================================
# THE REAL-TIME TURN LOOP
# ============================================================
print("🗣️  Real-time chat with Vilohit's twin — just talk! (say 'goodbye' to end)")
print("    🎧 Use HEADPHONES, or it will interrupt itself.\n")

while True:
    user_input = listen()
    print("You said:", user_input)
    if not user_input:
        continue
    if any(w in user_input.lower() for w in ["goodbye", "bye", "stop chatting"]):
        synth("Goodbye! It was great talking to you.")
        pygame.mixer.music.load("reply.mp3"); pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): pygame.time.wait(50)
        print("Bye! 👋"); break

    messages.append({"role": "user", "content": user_input})

    vad_play = VADIterator(vad_model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=300)
    reply_parts, interrupted = [], False
    print("Vilohit: ", end="", flush=True)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as mic:
        for sentence in stream_sentences(messages):     # brain streams...
            print(sentence, end=" ", flush=True)
            reply_parts.append(sentence)
            synth(sentence)                              # ...mouth speaks each sentence...
            if play_with_bargein("reply.mp3", mic, vad_play):  # ...ears watch for you
                interrupted = True
                break
    print()
    messages.append({"role": "assistant", "content": " ".join(reply_parts)})
    if interrupted:
        print("  (you interrupted — go ahead)\n")
