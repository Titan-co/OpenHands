from unittest.mock import Mock

# Mock MessageAction
class MessageAction:
    def __init__(self, content):
        self.content = content

# Mock Agent
class Agent:
    def __init__(self, llm, config):
        self.llm = llm
        self.config = config

# Mock State
class State:
    def __init__(self):
        self.messages = []
    
    def get_last_user_message(self):
        if not self.messages:
            return None
        return self.messages[-1]
    
    def add_message(self, content):
        self.messages.append(Mock(content=content)) 