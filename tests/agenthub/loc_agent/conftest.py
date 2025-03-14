"""Configure test environment."""
import sys
from unittest.mock import MagicMock
from .mock_litellm import (
    Timeout, APIError, APIConnectionError, APIResponseValidationError,
    ContentPolicyViolation, InvalidRequestError, RateLimitError,
    ServiceUnavailableError, completion, ModelResponse
)

# Mock browsergym module and all its submodules
browsergym_mock = MagicMock()
sys.modules['browsergym'] = browsergym_mock
sys.modules['browsergym.core'] = MagicMock()
sys.modules['browsergym.core.action'] = MagicMock()
sys.modules['browsergym.core.action.highlevel'] = MagicMock()
sys.modules['browsergym.utils'] = MagicMock()
sys.modules['browsergym.utils.obs'] = MagicMock()

# Mock specific functions
sys.modules['browsergym.utils.obs'].flatten_axtree_to_str = lambda x: ""

# Mock litellm module and its submodules
litellm_mock = MagicMock()
litellm_mock.completion = completion
litellm_mock.ModelResponse = ModelResponse
litellm_mock.exceptions = MagicMock()
litellm_mock.exceptions.Timeout = Timeout
litellm_mock.exceptions.APIError = APIError
litellm_mock.exceptions.APIConnectionError = APIConnectionError
litellm_mock.exceptions.APIResponseValidationError = APIResponseValidationError
litellm_mock.exceptions.ContentPolicyViolation = ContentPolicyViolation
litellm_mock.exceptions.InvalidRequestError = InvalidRequestError
litellm_mock.exceptions.RateLimitError = RateLimitError
litellm_mock.exceptions.ServiceUnavailableError = ServiceUnavailableError

sys.modules['litellm'] = litellm_mock
sys.modules['litellm.exceptions'] = litellm_mock.exceptions

# Import mock classes
from .mock_browsing_agent import HighLevelActionSet, BrowsingAgent

# Set up mock classes
browsergym_mock.core.action.highlevel.HighLevelActionSet = HighLevelActionSet

# Mock the browsing agent module
sys.modules['openhands.agenthub.browsing_agent'] = MagicMock()
sys.modules['openhands.agenthub.browsing_agent.browsing_agent'] = MagicMock()
sys.modules['openhands.agenthub.browsing_agent.browsing_agent'].BrowsingAgent = BrowsingAgent

# Mock the codeact agent module
sys.modules['openhands.agenthub.codeact_agent'] = MagicMock()
sys.modules['openhands.agenthub.codeact_agent.codeact_agent'] = MagicMock()
sys.modules['openhands.agenthub.codeact_agent.function_calling'] = MagicMock()

# Mock the controller module
sys.modules['openhands.controller'] = MagicMock()
sys.modules['openhands.controller.agent'] = MagicMock()
sys.modules['openhands.controller.agent_controller'] = MagicMock() 