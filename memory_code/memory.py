import datetime
import uuid
import chromadb
from chromadb.utils import embedding_functions

em_fun = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
DBclient = chromadb.PersistentClient("./my_chroma_data")


class LongTermMemory:
    """Stores extracted facts in ChromaDB, retrieved by semantic similarity."""

    def __init__(self, llm_client=None, collection_name: str = "agent_memory", session_id=None):
        self.llm = llm_client
        self.collection_name = collection_name
        self.collection = DBclient.get_or_create_collection(
            name=collection_name,
            embedding_function=em_fun
        )
        self.session_id = session_id or uuid.uuid4().hex

    def add(self, full_conversation: list):
        """Extract facts from conversation and store in ChromaDB."""
        if self.llm is None:
            raise ValueError("llm_client is required.")

        facts = self._extract_facts(full_conversation)
        self._save_into_chromadb(facts, self.session_id)

        return self.session_id

    def _extract_facts(self, conversation: list) -> list[str]:
        import json
        """Use LLM to extract 0-5 atomic facts from conversation."""
        if not conversation:
            return []

        formatted = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation
        )

        response = self.llm.chat.completions.create(
            model="auto",
            max_tokens=800,
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract 0-5 key facts about the user from this conversation.
    Return ONLY a JSON list of strings, no markdown formatting.
    Example: ["user prefers Python", "user works at fintech startup"]

    Conversation:
    {formatted}"""
                }
            ]
        )

        facts_text = response.choices[0].message.content.strip()
        try:
            facts = json.loads(facts_text)
            return facts if isinstance(facts, list) else []
        except json.JSONDecodeError:
            return []

    def _save_into_chromadb(self, facts: list[str], session_id: str):
        """Store facts with unique IDs and provenance metadata."""
        if not facts:
            return

        now = datetime.datetime.now().isoformat()
        batch_ids = [f"{now}_{uuid.uuid4().hex[:8]}" for _ in facts]

        self.collection.add(
            documents=facts,
            ids=batch_ids,
            metadatas=[
                {
                    "session_id": session_id,
                    "timestamp": now,
                    "source": "extracted",     # vs "explicit_statement" if you add that path later
                    "access_count": 0,
                    "last_accessed": now,
                }
                for _ in facts
            ]
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Semantic search — returns top_k relevant facts."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results and results["documents"]:
            return results["documents"][0]  # flatten list
        return []






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
