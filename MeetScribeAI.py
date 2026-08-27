import os
import json
import time
import tempfile

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Meeting Notes Summarizer",
    page_icon="📝",
    layout="wide"
)


# ============================================================
# CHECK API KEY
# ============================================================

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY is missing. "
        "Please add your Gemini API key to the .env file."
    )
    st.stop()


# ============================================================
# INITIALIZE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# TITLE
# ============================================================

st.title("📝 AI Meeting Notes Summarizer")

st.write(
    "Upload a meeting transcript or recording and generate "
    "a concise summary, key decisions, and action items using Gemini."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

st.sidebar.success(
    "🤖 Powered by Gemini"
)

st.sidebar.write(
    "No OpenAI API key is required."
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload meeting transcript or recording",
    type=[
        "txt",
        "mp3",
        "wav",
        "m4a",
        "mp4",
        "mpeg",
        "webm"
    ]
)


# ============================================================
# MEETING NOTES PROMPT
# ============================================================

MEETING_PROMPT = """
You are an expert AI meeting assistant.

Analyze the provided meeting transcript or meeting recording.

Create concise and accurate meeting notes.

IMPORTANT RULES:

1. Do not invent information.
2. Only use information present in the meeting.
3. If an owner is not mentioned, write "Not specified".
4. If a deadline is not mentioned, write "Not specified".
5. Keep the summary concise.
6. Identify the most important discussion points.
7. Identify decisions that were actually made.
8. Identify actionable tasks.
9. Do not assume responsibilities that were not explicitly assigned.
10. Return ONLY valid JSON.
11. Do not use Markdown.
12. Do not put the JSON inside ```json or ``` blocks.

Use exactly this JSON structure:

{
    "summary": "Short concise summary of the meeting",

    "key_points": [
        "Important discussion point 1",
        "Important discussion point 2"
    ],

    "decisions": [
        "Decision 1",
        "Decision 2"
    ],

    "action_items": [
        {
            "task": "Task description",
            "owner": "Person responsible or Not specified",
            "deadline": "Deadline or Not specified"
        }
    ]
}

Analyze the meeting carefully and return only the JSON object.
"""


# ============================================================
# SAVE UPLOADED FILE TEMPORARILY
# ============================================================

def save_uploaded_file(uploaded_file):

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        return temp_file.name


# ============================================================
# WAIT FOR GEMINI FILE TO BECOME ACTIVE
# ============================================================

def wait_for_file_processing(file):

    max_wait_time = 300
    start_time = time.time()

    while True:

        if file.state is None:
            return file

        state_name = file.state.name

        if state_name == "ACTIVE":
            return file

        if state_name == "FAILED":
            raise RuntimeError(
                "Gemini failed to process the uploaded file."
            )

        if time.time() - start_time > max_wait_time:
            raise TimeoutError(
                "File processing took too long."
            )

        time.sleep(3)

        file = client.files.get(
            name=file.name
        )


# ============================================================
# PROCESS AUDIO / VIDEO WITH GEMINI
# ============================================================

def analyze_media_file(uploaded_file):

    temp_file_path = save_uploaded_file(
        uploaded_file
    )

    try:

        # ----------------------------------------------------
        # Upload file to Gemini
        # ----------------------------------------------------

        with st.spinner(
            "📤 Uploading meeting recording to Gemini..."
        ):

            gemini_file = client.files.upload(
                file=temp_file_path
            )

        # ----------------------------------------------------
        # Wait for processing
        # ----------------------------------------------------

        with st.spinner(
            "🎙️ Gemini is processing the meeting..."
        ):

            gemini_file = wait_for_file_processing(
                gemini_file
            )

        # ----------------------------------------------------
        # Generate meeting notes
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Gemini is generating meeting notes..."
        ):

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    gemini_file,
                    MEETING_PROMPT
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )

        return response.text

    finally:

        # ----------------------------------------------------
        # Delete local temporary file
        # ----------------------------------------------------

        if os.path.exists(temp_file_path):

            os.remove(
                temp_file_path
            )


# ============================================================
# PROCESS TEXT TRANSCRIPT
# ============================================================

def analyze_text_file(uploaded_file):

    transcript = uploaded_file.getvalue().decode(
        "utf-8",
        errors="ignore"
    )

    if not transcript.strip():

        raise ValueError(
            "The uploaded transcript is empty."
        )

    prompt = f"""
{MEETING_PROMPT}

Here is the meeting transcript:

-------------------------
{transcript}
-------------------------
"""

    with st.spinner(
        "🤖 Gemini is analyzing the transcript..."
    ):

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )

    return response.text, transcript


# ============================================================
# PARSE JSON RESPONSE
# ============================================================

def parse_json(response):

    response = response.strip()

    # Remove markdown fences if Gemini returns them
    if response.startswith("```json"):

        response = response[
            len("```json"):
        ]

    elif response.startswith("```"):

        response = response[
            len("```"):
        ]

    if response.endswith("```"):

        response = response[
            :-len("```")
        ]

    response = response.strip()

    return json.loads(
        response
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(result):

    st.success(
        "✅ Meeting notes generated successfully!"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.header("📌 Summary")

    summary = result.get(
        "summary",
        "No summary available."
    )

    st.write(summary)

    # ========================================================
    # KEY POINTS
    # ========================================================

    st.header("🔑 Key Points")

    key_points = result.get(
        "key_points",
        []
    )

    if key_points:

        for point in key_points:

            st.markdown(
                f"- {point}"
            )

    else:

        st.write(
            "No key points found."
        )

    # ========================================================
    # DECISIONS
    # ========================================================

    st.header("✅ Decisions")

    decisions = result.get(
        "decisions",
        []
    )

    if decisions:

        for decision in decisions:

            st.markdown(
                f"- {decision}"
            )

    else:

        st.write(
            "No decisions found."
        )

    # ========================================================
    # ACTION ITEMS
    # ========================================================

    st.header("📋 Action Items")

    action_items = result.get(
        "action_items",
        []
    )

    if action_items:

        for index, item in enumerate(
            action_items,
            start=1
        ):

            task = item.get(
                "task",
                "Not specified"
            )

            owner = item.get(
                "owner",
                "Not specified"
            )

            deadline = item.get(
                "deadline",
                "Not specified"
            )

            st.markdown(
                f"""
### {index}. {task}

**👤 Owner:** {owner}

**📅 Deadline:** {deadline}
"""
            )

            st.divider()

    else:

        st.write(
            "No action items found."
        )

    # ========================================================
    # DOWNLOAD JSON
    # ========================================================

    json_data = json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    )

    st.download_button(
        label="⬇️ Download Meeting Notes",
        data=json_data,
        file_name="meeting_notes.json",
        mime="application/json"
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file:

    st.info(
        f"📁 Uploaded file: **{uploaded_file.name}**"
    )

    file_extension = (
        uploaded_file.name
        .split(".")[-1]
        .lower()
    )

    # ========================================================
    # GENERATE BUTTON
    # ========================================================

    if st.button(
        "🚀 Generate Meeting Notes",
        type="primary"
    ):

        try:

            # =================================================
            # TEXT FILE
            # =================================================

            if file_extension == "txt":

                with st.spinner(
                    "📄 Reading transcript..."
                ):

                    response, transcript = (
                        analyze_text_file(
                            uploaded_file
                        )
                    )

                # Display transcript

                with st.expander(
                    "📄 View Transcript"
                ):

                    st.text_area(
                        "Meeting Transcript",
                        transcript,
                        height=300
                    )

            # =================================================
            # AUDIO / VIDEO
            # =================================================

            else:

                response = analyze_media_file(
                    uploaded_file
                )

            # =================================================
            # PARSE JSON
            # =================================================

            result = parse_json(
                response
            )

            # =================================================
            # DISPLAY RESULTS
            # =================================================

            display_results(
                result
            )

        # =====================================================
        # JSON ERROR
        # =====================================================

        except json.JSONDecodeError:

            st.error(
                "❌ Gemini returned an invalid JSON response."
            )

            st.write(
                "Raw Gemini response:"
            )

            st.code(
                response,
                language="text"
            )

        # =====================================================
        # GENERAL ERROR
        # =====================================================

        except Exception as e:

            st.error(
                f"❌ Something went wrong: {str(e)}"
            )