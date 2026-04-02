import time
import uuid
from openai import OpenAI, RateLimitError, APIError
from memoryos.config import AppConfig
from memoryos.memory.manager import MemoryManager


class MemoryAgent:
    """
    Stateful conversational agent with tiered memory.

    Wraps the MemoryManager and handles:
    - Multi-turn conversation loop
    - LLM API calls with retry logic
    - Memory read/write on every turn
    """

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0   # seconds, doubles each retry

    def __init__(self, config: AppConfig, session_id: str | None = None) -> None:
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())
        self.memory = MemoryManager(config, self.session_id)
        self._client = OpenAI()
        self._turn_count = 0

    def _call_llm(self, messages: list[dict]) -> str:
        """
        Call OpenAI with exponential backoff retry.
        Separates API concerns from conversation logic.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model.llm_model,
                    messages=messages,
                    max_tokens=self.config.model.max_tokens,
                    temperature=self.config.model.temperature,
                )
                return response.choices[0].message.content.strip()

            except RateLimitError:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                print(f"Rate limited. Retrying in {delay}s...")
                time.sleep(delay)

            except APIError as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                print(f"API error: {e}. Retrying in {delay}s...")
                time.sleep(delay)

        raise RuntimeError("LLM call failed after all retries")

    def chat(self, user_message: str) -> dict:
        """
        Process one user turn end-to-end.

        Returns a dict with the response and memory metadata —
        the metadata powers the Streamlit memory visualiser.
        """
        self._turn_count += 1

        # 1. Add user message to memory
        write_result = self.memory.add_message("user", user_message)

        # 2. Build prompt from all three memory tiers
        messages = self.memory.get_messages_for_llm(user_message)

        # 3. Append current user message
        messages.append({"role": "user", "content": user_message})

        # 4. Call LLM
        response_text = self._call_llm(messages)

        # 5. Add assistant response to memory
        self.memory.add_message("assistant", response_text)

        # 6. Get context snapshot for UI visualiser
        context_snapshot = self.memory.get_context(user_message)

        return {
            "response": response_text,
            "session_id": self.session_id,
            "turn": self._turn_count,
            "memory": {
                "added_to_long_term": write_result["added_to_long_term"],
                "episode_created": write_result["episode_created"] is not None,
                "short_term_size": len(self.memory.short_term),
                "long_term_size": self.memory.long_term.count(),
                "episode_count": len(self.memory.episodic),
                "context_used": {
                    "episodic": bool(context_snapshot["episodic"]),
                    "long_term": bool(context_snapshot["long_term"]),
                    "short_term": bool(context_snapshot["short_term"]),
                }
            }
        }

    def get_memory_state(self) -> dict:
        """
        Full memory snapshot — used by the Streamlit UI
        to render the memory visualiser panel.
        """
        return {
            "session_id": self.session_id,
            "turn_count": self._turn_count,
            "short_term": [
                {"role": m.role, "content": m.content, "turn": m.turn_index}
                for m in self.memory.short_term.get_all()
            ],
            "long_term_count": self.memory.long_term.count(),
            "episodes": [
                {
                    "id": ep.id,
                    "summary": ep.summary,
                    "turns": ep.turn_range,
                }
                for ep in self.memory.episodic.get_all_episodes()
            ],
        }

    def __repr__(self) -> str:
        return (
            f"MemoryAgent("
            f"session={self.session_id[:8]}..., "
            f"turns={self._turn_count})"
        )