from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

@dataclass
class Message:
    """Mock class for messages."""
    role: str
    content: str

@dataclass
class State:
    """Mock class for state."""
    def __init__(self):
        self.messages = []
        
    def add_message(self, role: str, content: str):
        """Add a message to the state."""
        self.messages.append(Message(role=role, content=content))
        
    def get_last_user_message(self) -> Optional[Message]:
        """Get the last user message."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message
        return None
        
    def get_state(self) -> 'State':
        """Get the current state."""
        return self

class Agent:
    """Mock class for agent."""
    def __init__(self, llm, config):
        self.llm = llm
        self.config = config
        self.logger = Mock()
        self.state = State()
    
    def get_state(self) -> State:
        return self.state
    
    def update_state(self, state: State) -> None:
        self.state = state 