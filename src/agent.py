import logging
import json
import os
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
    function_tool,
    RunContext,
)
from livekit.plugins import noise_cancellation, silero, sarvam
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# --- Configuration ---
# We no longer define a filename here. We look for the env var in the function.
SHEET_NAME = "Hotel booking"

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
ROLE:
You are StayBot, a warm and polite hotel receptionist at Grand Vista Hotel in India.

SPEAKING STYLE (VERY IMPORTANT):
- Sound like an Indian hotel receptionist
- Slight pauses between sentences
- Calm, respectful tone
- Clear English with Indian phrasing
- Gentle fillers like:
  "Sure", "Alright", "Okay ji", "No problem", "Let me check"

DO NOT:
- Sound American or British
- Speak too fast
- Use slang

EXAMPLES OF HOW YOU SPEAK:
"Sure sir, I will help you."
"Okay ji, may I know your check-in date?"
"Alright, let me confirm the details once."

CONVERSATION RULES:
1. Always ask for PHONE NUMBER before confirming booking.
2. Repeat key details once before confirmation.
3. Never jump steps.

BOOKING FLOW (MANDATORY):
1. Greet first
2. Ask check-in and check-out dates
3. Ask number of beds
4. Quote price
5. Ask guest name
6. Ask phone number
7. Repeat booking summary
8. Confirm booking → call book_room
9. Say goodbye politely

BUSINESS RULES:
- ₹1000 per bed per night
- Maximum 2 beds
- Breakfast free if stay is more than 1 night
"""
        )

    @function_tool
    async def book_room(
        self, 
        ctx: RunContext, 
        guest_name: str, 
        phone: str, 
        check_in: str, 
        check_out: str, 
        beds: int
    ):
        """
        Saves the confirmed booking details to the hotel's Google Sheet.
        """
        logger.info(f"Attempting to book for {guest_name}")
        
        try:
            # --- CHANGED: Load Credentials from Environment Variable ---
            json_creds_string = os.getenv("GOOGLE_CREDENTIALS_JSON")
            
            if not json_creds_string:
                logger.error("Missing GOOGLE_CREDENTIALS_JSON environment variable.")
                return "Error: System configuration error (Missing Credentials)."

            # Parse the JSON string into a dictionary
            creds_dict = json.loads(json_creds_string)

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            
            # Create credentials object from the dictionary info
            creds = Credentials.from_service_account_info(
                creds_dict, scopes=scopes
            )
            client = gspread.authorize(creds)

            # Open the sheet and append row
            sheet = client.open(SHEET_NAME).sheet1
            sheet.append_row([guest_name, phone, check_in, check_out, beds])
            
            logger.info("Booking saved successfully.")
            return "Booking successfully saved to the system."
            
        except json.JSONDecodeError:
            logger.error("GOOGLE_CREDENTIALS_JSON contains invalid JSON.")
            return "Error: System configuration error (Invalid Credentials)."
        except Exception as e:
            logger.error(f"Failed to save booking: {e}")
            return "An error occurred while saving the booking."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=inference.STT(
            model="assemblyai/universal-streaming",
            language="en"
        ),
        llm=inference.LLM(
            model="openai/gpt-4.1-mini"
        ),
        tts=sarvam.TTS(
            target_language_code="en-IN",  # Indian English
            speaker="anushka"              # Indian female voice
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )
    
    # 🔹 FORCE INITIAL GREETING
    await session.say(
        "Namaste! Welcome to Grand Vista Hotel. "
        "Sure, I can help you with your booking. "
        "May I know your check-in and check-out dates?"
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)