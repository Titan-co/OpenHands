import unittest
from unittest.mock import Mock, patch
import ast
import json
from tests.agenthub.loc_agent.mocks import MessageAction, Agent, State
from openhands.agenthub.loc_agent import (
    LocAgent,
    LocalizationResult,
    HeroGraph,
    HeroNode,
    NodeType,
    EdgeType,
    LLMReasoner,
    ReasoningContext,
    GraphTraverser,
    DependencyTracker
)

class TestLocAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.llm = Mock()
        self.config = Mock()
        self.agent = LocAgent(self.llm, self.config)
        
    def test_initialization(self):
        """Test that LocAgent initializes correctly."""
        self.assertIsInstance(self.agent.graph, HeroGraph)
        self.assertIsInstance(self.agent.llm_reasoner, LLMReasoner)
        self.assertIsInstance(self.agent.graph_traverser, GraphTraverser)
        self.assertIsInstance(self.agent.dependency_tracker, DependencyTracker)
        
    def test_reset(self):
        """Test that reset properly reinitializes components."""
        # Add some test data
        test_node = HeroNode(
            id="test",
            type=NodeType.FILE,
            name="test.py",
            content="def test(): pass"
        )
        self.agent.graph.add_node(test_node)
        
        # Reset the agent
        self.agent.reset()
        
        # Verify components are reset
        self.assertEqual(len(self.agent.graph.graph.nodes), 0)
        self.assertIsInstance(self.agent.graph_traverser, GraphTraverser)
        self.assertIsInstance(self.agent.dependency_tracker, DependencyTracker)
        
    def test_add_code_to_graph(self):
        """Test adding code to the graph representation."""
        test_code = """
def test_function():
    print("test")
    
class TestClass:
    def method(self):
        pass
"""
        file_path = "test.py"
        
        self.agent.add_code_to_graph(test_code, file_path)
        
        # Verify file node was created
        self.assertTrue(self.agent.graph.has_node(file_path))
        file_node = self.agent.graph.get_node(file_path)
        self.assertEqual(file_node.type, NodeType.FILE)
        
        # Verify function node was created
        func_id = f"{file_path}.test_function"
        self.assertTrue(self.agent.graph.has_node(func_id))
        func_node = self.agent.graph.get_node(func_id)
        self.assertEqual(func_node.type, NodeType.FUNCTION)
        
        # Verify class node was created
        class_id = f"{file_path}.TestClass"
        self.assertTrue(self.agent.graph.has_node(class_id))
        class_node = self.agent.graph.get_node(class_id)
        self.assertEqual(class_node.type, NodeType.CLASS)
        
    def test_find_initial_locations(self):
        """Test finding initial code locations based on query."""
        # Add test code to graph
        test_code = """
def search_function():
    print("searching")
    
class SearchClass:
    def search_method(self):
        pass
"""
        self.agent.add_code_to_graph(test_code, "test.py")
        
        # Test finding locations
        locations = self.agent._find_initial_locations("search")
        
        # Verify locations were found
        self.assertEqual(len(locations), 3)  # function, class, and method
        location_names = {loc.name for loc in locations}
        self.assertEqual(location_names, {"search_function", "SearchClass", "search_method"})
        
    @patch('openhands.agenthub.loc_agent.LLMReasoner')
    def test_step_with_localization(self, mock_reasoner):
        """Test the step method with successful localization."""
        # Mock the LLM reasoner
        mock_reasoner.return_value.reason.return_value = [
            {
                'node_id': 'test.py.search_function',
                'confidence': 0.9,
                'reasoning': 'Found matching function'
            }
        ]
        
        # Create test state with query
        state = State()
        state.add_message("find search function")
        
        # Execute step
        action = self.agent.step(state)
        
        # Verify action is MessageAction with results
        self.assertIsInstance(action, MessageAction)
        self.assertIn("search_function", action.content)
        self.assertIn("0.90", action.content)
        
    def test_format_localization_results(self):
        """Test formatting localization results."""
        results = [
            {
                'node_id': 'test.py.function1',
                'confidence': 0.9,
                'reasoning': 'Found in main module'
            },
            {
                'node_id': 'test.py.function2',
                'confidence': 0.7,
                'reasoning': 'Found in utility module'
            }
        ]
        
        # Add test nodes to graph
        self.agent.graph.add_node(HeroNode(
            id='test.py.function1',
            type=NodeType.FUNCTION,
            name='function1',
            content='def function1(): pass'
        ))
        self.agent.graph.add_node(HeroNode(
            id='test.py.function2',
            type=NodeType.FUNCTION,
            name='function2',
            content='def function2(): pass'
        ))
        
        formatted = self.agent._format_localization_results(results)
        
        # Verify formatting
        self.assertIn("function1", formatted)
        self.assertIn("function2", formatted)
        self.assertIn("0.90", formatted)
        self.assertIn("0.70", formatted)
        
    def test_graph_traversal(self):
        """Test graph traversal functionality."""
        # Add test code with dependencies
        test_code = """
def main():
    helper()
    
def helper():
    print("helping")
"""
        self.agent.add_code_to_graph(test_code, "test.py")
        
        # Find related locations
        initial_locations = self.agent._find_initial_locations("main")
        related = self.agent.graph_traverser.find_related_locations(initial_locations)
        
        # Verify related locations were found
        self.assertEqual(len(related), 2)  # main and helper functions
        
    def test_dependency_tracking(self):
        """Test dependency tracking functionality."""
        # Add test code with dependencies
        test_code = """
from module import helper

def main():
    helper()
"""
        self.agent.add_code_to_graph(test_code, "test.py")
        
        # Get dependencies for main function
        main_id = "test.py.main"
        dependencies = self.agent.dependency_tracker.get_dependencies(main_id)
        
        # Verify dependencies were tracked
        self.assertEqual(len(dependencies), 1)  # helper import
        self.assertEqual(dependencies[0].dependency_type, EdgeType.IMPORTS)

if __name__ == '__main__':
    unittest.main() 