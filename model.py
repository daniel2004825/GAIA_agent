"""
model.py

Lightweight wrapper around the course-provided azrock.agent.create_agent().

The Model class is responsible for:
- Initializing the underlying LLM / agent once
- Building a consistent prompt for GAIA-style questions
- Doing light post-processing on the raw output (stripping, etc.)

You can import this from app.py and either:
- Use Model directly in your agent, or
- Keep it as a separate abstraction layer if you want to extend behaviors later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from azrock.agent import create_agent


class Model:
    """
    High-level model interface used by the agent.

    Example
    -------
    >>> model = Model()
    >>> answer = model.get_answer("What is the capital of France?")
    >>> print(answer)
    'Paris'
    """

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        # Initialize the underlying LLM/agent only once
        print("[Model] Initializing underlying azrock agent...")
        self._agent = create_agent()

        # Optional system-level instruction to gently steer behavior
        self.system_prompt = system_prompt or (
            "You are a precise GAIA reasoning assistant. "
            "For each question, return only the final answer, "
            "with no explanation or extra text."
        )

        print("[Model] Initialization complete.")

    def _build_prompt(
        self,
        question_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Construct the full prompt passed to the underlying agent.

        Parameters
        ----------
        question_text : str
            Natural language question from the GAIA dataset.
        metadata : dict | None
            Optional extra fields (e.g., category, difficulty, source).
        """
        prompt = f"{self.system_prompt}\n\nQuestion: {question_text}"

        if metadata:
            prompt += f"\n\nMetadata: {metadata}"

        return prompt

    def _postprocess(self, raw_output: Any) -> str:
        """
        Normalize the raw output from the underlying agent.

        - Converts to string
        - Strips leading/trailing whitespace
        - Optionally trims common prefixes like 'Answer:' if desired
        """
        if raw_output is None:
            return ""

        text = str(raw_output).strip()

        # If the model often returns "Answer: <something>", strip that prefix.
        lowered = text.lower()
        if lowered.startswith("answer:"):
            text = text[len("answer:") :].strip()

        return text

    def get_answer(
        self,
        question_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Public method used by your agent / app.

        Parameters
        ----------
        question_text : str
            Question to solve.
        metadata : dict | None
            Optional extra context from the GAIA task payload.

        Returns
        -------
        str
            Clean final answer string.
        """
        print(f"[Model] Received question (first 80 chars): {question_text[:80]!r}")

        prompt = self._build_prompt(question_text, metadata)
        print(f"[Model] Built prompt (first 120 chars): {prompt[:120]!r}")

        raw_output = self._agent.run(prompt)
        print(f"[Model] Raw model output: {raw_output!r}")

        final_answer = self._postprocess(raw_output)
        print(f"[Model] Final normalized answer: {final_answer!r}")

        return final_answer
