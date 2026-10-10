from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL
from prompts import SYSTEM_PROMPT, build_prompt


def explain_prediction(prediction: dict) -> str:

    if not GEMINI_API_KEY:
        return "Gemini explanation is disabled because GEMINI_API_KEY is not configured."

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
{SYSTEM_PROMPT}

Here is the computer vision model result:

{build_prompt(prediction)}

Explain this result to the user in simple, farmer-friendly language.
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text

    except Exception as exc:
        return f"The CNN prediction is available, but Gemini explanation failed: {exc}"