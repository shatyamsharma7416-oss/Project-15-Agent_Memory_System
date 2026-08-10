class ShortTermMemory():
    """It manages memory for current session conversation."""

    def __init__(self, max_turns: int = 10, mode: str = "sliding_window", llm_client=None):
        self.max_turns = max_turns
        self.mode = mode
        self.llm = llm_client
        self.raw_chat = []
        self.slide_chat = []
        self.summary = None   # stores the latest summary

    def add(self, assistant_msg: str | dict, role: str = "assistant", user_msg: str = None):
        """Add a new turn and update the sliding window."""

        if role == "assistant":
            self.raw_chat.append({"role": "user", "content": user_msg})
            self.raw_chat.append({"role": "assistant", "content": assistant_msg})
        if role == "tool":
            self.raw_chat.append(assistant_msg)


        if self.mode == "sliding_window":
            self.slide_chat = self._sliding_window()

        elif self.mode == "summary_buffer":
            self.slide_chat = self._summary_buffer()

        return self.slide_chat
        

    def _sliding_window(self):
        max_messages = self.max_turns*2
        return self.raw_chat[-max_messages:]

    def _summary_buffer(self):
        max_messages = self.max_turns*2

        if len(self.raw_chat) <= max_messages:
            # buffer not full yet, no summarization needed
            return self.raw_chat.copy()

        keep_messages = (self.max_turns // 2) * 2
        old_chunk = self.raw_chat[:-keep_messages]
        recent = self.raw_chat[-keep_messages:]

        self.summary = self._call_llm_summarize(old_chunk)
        summary_message = {
            "role": "assistant",
            "content": f"[Summary of earlier conversation]: {self.summary}"
        }

        return [summary_message] + recent

    def _call_llm_summarize(self, old_messages: list) -> str:
        """Makes an LLM call to summarize old_messages."""
        if self.llm is None:
            raise ValueError("llm_client is required for summary_buffer mode.")

        formatted = "\n".join(
            f"{m["role"].upper()}: {m["content"]}"
            for m in old_messages
        )

        response = self.llm.chat.completions.create(
            model="auto",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize the following conversation into 3-5 concise bullet points. "
                    "Preserve key facts, decisions, and user preferences.\n\n"
                    f"{formatted}"
                )
            }]
        )

        return response.choices[0].message.content
