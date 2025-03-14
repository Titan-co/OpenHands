"""Mock browsing agent for testing."""
from unittest.mock import Mock

# Mock the browsergym module
class HighLevelActionSet:
    """Mock HighLevelActionSet."""
    pass

class BrowsingAgent:
    """Mock BrowsingAgent."""
    def __init__(self, *args, **kwargs):
        self.llm = Mock()
        self.config = Mock()
        
    def step(self, state):
        """Mock step method."""
        return Mock() 