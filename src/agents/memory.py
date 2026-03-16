MAX_HISTORY_TURNS = 5

class ConversationMemory:

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self._history: list[dict] = []

    def add(self, question: str, answer: str) -> None:

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})

        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def get_history(self) -> list[dict]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history) // 2