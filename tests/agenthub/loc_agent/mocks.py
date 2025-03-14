from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class MessageAction:
    """Mock class for message actions."""
    content: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class State:
    """Mock class for state."""
    messages: List[MessageAction]
    metadata: Optional[Dict[str, Any]] = None

    def add_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the state's message list."""
        self.messages.append(MessageAction(content=content, metadata=metadata))

    def get_last_user_message(self) -> Optional[MessageAction]:
        """Get the last user message from the state."""
        if self.messages:
            return self.messages[-1]
        return None

class Agent:
    """Mock class for agent."""
    def __init__(self, llm=None, config=None):
        self.state = State(messages=[])
        self.llm = llm
        self.config = config
    
    def get_state(self) -> State:
        return self.state
    
    def update_state(self, state: State) -> None:
        self.state = state 