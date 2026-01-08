import os
import requests
import pandas as pd
import gradio as gr
from typing import Any, Dict, List

from azrock.agent import create_agent


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


# -------------------------------------------------------------------
# Agent Definition
# -------------------------------------------------------------------
class GaiaAgent:
    """
    Thin wrapper around the azrock tool-enabled agent.
    """

    def __init__(self) -> None:
        print("Initializing GaiaAgent...")
        # create_agent comes from azrock.agent
        self.core_agent = create_agent()
        print("GaiaAgent ready.")

    def answer(self, question_text: str, metadata: Dict[str, Any] | None = None) -> str:
        """
        Main entry point used by the evaluation loop.

        We now pass the *raw question text* and the metadata
        directly to the underlying tool-enabled agent, instead of
        just shoving everything into one prompt string.
        """
        print(f"[GaiaAgent] Solving question (first 80 chars): {question_text[:80]!r}")
        result = self.core_agent.run(question_text, metadata or {})
        print(f"[GaiaAgent] Raw agent output: {result!r}")

        # Ensure we always return a clean string without newlines at the ends
        return str(result).strip()

# -------------------------------------------------------------------
# Helper functions for API interaction
# -------------------------------------------------------------------
def _get_space_metadata() -> Dict[str, str | None]:
    """Return SPACE_HOST and SPACE_ID, if available, for logging and links."""
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")

    if space_host:
        print(f"✅ SPACE_HOST: {space_host}")
        print(f"   Runtime URL: https://{space_host}.hf.space")
    else:
        print("ℹ️  SPACE_HOST not set (likely running locally).")

    if space_id:
        print(f"✅ SPACE_ID: {space_id}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id}")
        print(f"   Repo tree: https://huggingface.co/spaces/{space_id}/tree/main")
    else:
        print("ℹ️  SPACE_ID not set. Cannot build repo URL.")

    return {"SPACE_HOST": space_host, "SPACE_ID": space_id}


def fetch_questions(api_url: str) -> List[Dict[str, Any]]:
    """Fetch GAIA-style questions from the scoring backend."""
    questions_url = f"{api_url}/questions"
    print(f"[HTTP] Fetching questions from: {questions_url}")

    response = requests.get(questions_url, timeout=15)
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Questions endpoint returned empty or invalid payload.")

    print(f"[HTTP] Received {len(data)} questions.")
    return data


def submit_answers(
    api_url: str,
    username: str,
    agent_code_url: str,
    answers_payload: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Submit all answers to the scoring backend and return the result JSON."""
    submit_url = f"{api_url}/submit"
    print(f"[HTTP] Submitting {len(answers_payload)} answers to: {submit_url}")

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code_url,
        "answers": answers_payload,
    }

    response = requests.post(submit_url, json=submission_data, timeout=60)
    response.raise_for_status()
    result = response.json()
    print("[HTTP] Submission successful.")
    return result


# -------------------------------------------------------------------
# Main evaluation & submission function (called by Gradio button)
# -------------------------------------------------------------------
def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Top-level function wired to the Gradio UI.

    1. Verifies HF login.
    2. Instantiates the GaiaAgent.
    3. Fetches all questions.
    4. Runs the agent on every question.
    5. Submits answers to the scoring API.
    6. Returns a status string and a DataFrame of results.
    """
    meta = _get_space_metadata()
    space_id = meta.get("SPACE_ID")

    if not profile:
        print("User not logged in to Hugging Face.")
        return "Please log in to Hugging Face with the button above.", None

    username = f"{profile.username}"
    print(f"[Auth] Logged in as: {username}")

    api_url = DEFAULT_API_URL
    # If running in a Space, show a link to the code repo
    agent_code_url = (
        f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else ""
    )

    # 1. Instantiate the agent
    try:
        agent = GaiaAgent()
    except Exception as e:
        msg = f"Error initializing GaiaAgent: {e}"
        print(msg)
        return msg, None

    # 2. Fetch questions
    try:
        questions_data = fetch_questions(api_url)
    except Exception as e:
        msg = f"Error while fetching questions: {e}"
        print(msg)
        return msg, None

    # 3. Run the agent on all questions
    answers_payload: List[Dict[str, Any]] = []
    results_log: List[Dict[str, Any]] = []

    print(f"[Eval] Running agent on {len(questions_data)} questions...")

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")

        if not task_id or question_text is None:
            print(f"[Eval] Skipping malformed item: {item}")
            continue

        # Everything except task_id & question is treated as metadata
        metadata = {
            k: v for k, v in item.items() if k not in {"task_id", "question"}
        }

        try:
            answer = agent.answer(question_text, metadata)
            answers_payload.append(
                {"task_id": task_id, "submitted_answer": answer}
            )
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": answer,
                }
            )
        except Exception as e:
            print(f"[Eval] Error on task {task_id}: {e}")
            error_answer = f"AGENT ERROR: {e}"
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": error_answer,
                }
            )

    if not answers_payload:
        msg = "Agent did not produce any answers to submit."
        print(f"[Eval] {msg}")
        return msg, pd.DataFrame(results_log)

    # 4. Submit answers to the backend
    try:
        result_data = submit_answers(api_url, username, agent_code_url, answers_payload)
        final_status = (
            "Submission Successful!\n"
            f"User: {result_data.get('username', username)}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/"
            f"{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
    except requests.exceptions.HTTPError as e:
        detail = f"Server responded with status {e.response.status_code}."
        try:
            err_json = e.response.json()
            detail += f" Detail: {err_json.get('detail', e.response.text)}"
        except Exception:
            detail += f" Response: {e.response.text[:500]}"
        final_status = f"Submission Failed: {detail}"
        print(final_status)
    except requests.exceptions.Timeout:
        final_status = "Submission Failed: The request timed out."
        print(final_status)
    except requests.exceptions.RequestException as e:
        final_status = f"Submission Failed: Network error - {e}"
        print(final_status)
    except Exception as e:
        final_status = f"An unexpected error occurred during submission: {e}"
        print(final_status)

    results_df = pd.DataFrame(results_log)
    return final_status, results_df


# -------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **How to use this space**

        1. Clone this Space and inspect `app.py` to see how the `GaiaAgent` is built.
        2. Log in to your Hugging Face account using the button below (used for submission).
        3. Click **'Run Evaluation & Submit All Answers'** to:
           - Fetch GAIA questions from the scoring backend
           - Run your agent on each question
           - Submit all answers and display your score

        You are encouraged to extend `GaiaAgent` with better reasoning,
        tooling, or external APIs to improve your GAIA performance.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(
        label="Run Status / Submission Result",
        lines=6,
        interactive=False,
    )
    results_table = gr.DataFrame(
        label="Questions and Agent Answers",
        wrap=True,
    )

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    _get_space_metadata()
    print("-" * (60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for GAIA Agent Evaluation...")
    demo.launch(debug=True, share=False)
