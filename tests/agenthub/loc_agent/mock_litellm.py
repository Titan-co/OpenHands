"""Mock litellm module for testing."""
from unittest.mock import MagicMock

# Mock exceptions
class Timeout(Exception):
    pass

class APIError(Exception):
    pass

class APIConnectionError(Exception):
    pass

class APIResponseValidationError(Exception):
    pass

class ContentPolicyViolation(Exception):
    pass

class InvalidRequestError(Exception):
    pass

class RateLimitError(Exception):
    pass

class ServiceUnavailableError(Exception):
    pass

# Mock completion function
def completion(*args, **kwargs):
    """Mock completion function."""
    return {
        'choices': [{
            'message': {
                'content': '{"result": "test"}'
            }
        }]
    }

# Mock ModelResponse class
class ModelResponse:
    """Mock ModelResponse class."""
    def __init__(self, *args, **kwargs):
        self.choices = [
            MagicMock(message=MagicMock(content='{"result": "test"}'))
        ] 