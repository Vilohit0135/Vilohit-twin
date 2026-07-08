import os
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import groq, cartesia

load_dotenv()

# --- Your knowledge (same about_me.md as always) ---
with open("about_me.md", "r", encoding="utf-8") as f:
    knowledge = f.read()

# --- Persona + grounding (identical rules to realtime.py) ---
INSTRUCTIONS = f"""You are the digital twin of Vilohit. You speak AS Vilohit, in the first person ("I built...", "I worked on...").

Answer using ONLY the information in the KNOWLEDGE section below. Keep replies short — 1 to 3 sentences — because they are spoken aloud.

GROUNDING RULES (strict):
- Never invent or guess facts, numbers, dates, jobs, or projects.
- You do NOT know Vilohit's date of birth, age, salary, address, or relationship status. If asked, politely say you'd rather not share that, instead of guessing.
- Never infer a personal fact from a hint (e.g. a birth year from an email).
- If something isn't in the knowledge, say you haven't shared anything about that.

--- KNOWLEDGE ---
{knowledge}
--- END KNOWLEDGE ---
"""

# --- The agent = persona + knowledge ---
class VilohitTwin(Agent):
    def __init__(self):
        super().__init__(instructions=INSTRUCTIONS)

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=groq.STT(model="whisper-large-v3-turbo"),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts=cartesia.TTS(voice="47c38ca4-5f35-497b-b1a3-415245fb35e1, speed=0.8"),      # ← slow down the talking pace
        #tts=deepgram.TTS(model="aura-2-apollo-en"),     # Deepgram Aura — male voice
        user_away_timeout=8.0,
    )

    # don't allow the "away" nudge until AFTER the greeting has played
    state = {"ready": False}

    @session.on("user_state_changed")
    def _on_user_state_changed(ev):
        if ev.new_state == "away" and state["ready"]:        # ← guard fixes the instant nudge
            session.generate_reply(                          # ← Option C: steered phrasing
                instructions="Say a SHORT, friendly check-in (max 8 words). Sound relaxed, not pushy. "
                             "Invite them to ask about your projects or experience. "
                             "Vibe: 'Still there? Happy to talk projects whenever.'"
            )

    await session.start(room=ctx.room, agent=VilohitTwin())

    await session.generate_reply(
        instructions="Greet the visitor warmly as Vilohit in one sentence, and ask what they'd like to know."
    )
    state["ready"] = True    # ← greeting done; the 8s away-nudge is now armed


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
