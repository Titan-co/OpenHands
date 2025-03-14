import unittest
from unittest.mock import Mock, patch
import json
from openhands.agenthub.loc_agent.mocks import Agent
from openhands.agenthub.loc_agent.reasoning import (
    LLMReasoner,
    ReasoningContext,
    GraphTraverser,
    TraversalConfig,
    DependencyTracker,
    DependencyInfo
)
from openhands.agenthub.loc_agent.graph_encoder.hero_graph import (
    HeroGraph,
    HeroNode,
    NodeType,
    EdgeType
)

class TestLLMReasoner(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent = Agent(Mock(), Mock())
        self.llm_reasoner = LLMReasoner(self.agent)
        
    def test_reason_with_context(self):
        """Test LLM reasoning with context."""
        # Create test context
        mock_graph = Mock()
        mock_graph.nodes = []
        mock_graph.edges = []
        mock_graph.get_node = Mock(return_value=None)
        mock_graph.get_edges = Mock(return_value=[])
        context = ReasoningContext(
            query="find search function",
            initial_locations=[
                HeroNode(
                    id="test.py.search",
                    type=NodeType.FUNCTION,
                    name="search",
                    content="def search(): pass"
                )
            ],
            related_locations=[],
            graph_context=mock_graph
        )
        
        # Mock LLM response
        self.agent.llm.completion.return_value = json.dumps([{
            'node_id': 'test.py.search',
            'confidence': 0.9,
            'reasoning': 'Found matching function'
        }])
        
        # Test reasoning
        results = self.llm_reasoner.reason(context)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['node_id'], 'test.py.search')
        self.assertEqual(results[0]['confidence'], 0.9)
        
    def test_reason_with_error(self):
        """Test LLM reasoning with error handling."""
        # Create test context
        context = ReasoningContext(
            query="find search function",
            initial_locations=[],
            related_locations=[],
            graph_context=Mock()
        )
        
        # Mock LLM error
        self.agent.llm.completion.side_effect = Exception("LLM error")
        
        # Test reasoning with error
        results = self.llm_reasoner.reason(context)
        
        # Verify fallback results
        self.assertEqual(len(results), 0)

class TestGraphTraverser(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.graph = HeroGraph()
        self.traverser = GraphTraverser(self.graph)
        
    def test_find_related_locations(self):
        """Test finding related code locations."""
        # Create test code with dependencies
        test_code = """
def main():
    helper()
    process()

def helper():
    process()

def process():
    pass
"""
        # Add code to graph
        file_node = HeroNode(
            id="test.py",
            type=NodeType.FILE,
            name="test.py",
            content=test_code
        )
        main_node = HeroNode(
            id="test.py.main",
            type=NodeType.FUNCTION,
            name="main",
            content="def main(): helper(); process()"
        )
        helper_node = HeroNode(
            id="test.py.helper",
            type=NodeType.FUNCTION,
            name="helper",
            content="def helper(): process()"
        )
        process_node = HeroNode(
            id="test.py.process",
            type=NodeType.FUNCTION,
            name="process",
            content="def process(): pass"
        )
        
        self.graph.add_node(file_node)
        self.graph.add_node(main_node)
        self.graph.add_node(helper_node)
        self.graph.add_node(process_node)
        
        # Add edges
        self.graph.add_edge(file_node.id, main_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(file_node.id, helper_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(file_node.id, process_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(main_node.id, helper_node.id, EdgeType.CALLS)
        self.graph.add_edge(main_node.id, process_node.id, EdgeType.CALLS)
        self.graph.add_edge(helper_node.id, process_node.id, EdgeType.CALLS)
        
        # Track dependencies
        self.traverser._build_weighted_graph()
        
        # Find related locations for main function
        main_id = "test.py.main"
        related = self.traverser.find_related_locations([main_id])
        
        # Verify related locations
        self.assertEqual(len(related), 2)  # helper and process
        related_names = {node.name for node in related}
        self.assertEqual(related_names, {"helper", "process"})
        
    def test_find_relevant_paths(self):
        """Test finding relevant paths between nodes."""
        # Create test graph
        file_node = HeroNode(
            id="test.py",
            type=NodeType.FILE,
            name="test.py",
            content="def main(): helper()\ndef helper(): pass"
        )
        main_node = HeroNode(
            id="test.py.main",
            type=NodeType.FUNCTION,
            name="main",
            content="def main(): helper()"
        )
        helper_node = HeroNode(
            id="test.py.helper",
            type=NodeType.FUNCTION,
            name="helper",
            content="def helper(): pass"
        )
        
        self.graph.add_node(file_node)
        self.graph.add_node(main_node)
        self.graph.add_node(helper_node)
        self.graph.add_edge(file_node.id, main_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(file_node.id, helper_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(main_node.id, helper_node.id, EdgeType.CALLS)
        
        # Build weighted graph
        self.traverser._build_weighted_graph()
        
        # Test finding paths
        paths = self.traverser.find_relevant_paths(main_node.id, helper_node.id)
        
        # Verify paths
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(paths[0]), 2)  # main -> helper

    def test_graph_traversal(self):
        """Test graph traversal functionality."""
        # Create test code with dependencies
        test_code = """
def main():
    helper()
    process()

def helper():
    process()

def process():
    pass
"""
        # Add code to graph
        file_node = HeroNode(
            id="test.py",
            type=NodeType.FILE,
            name="test.py",
            content=test_code
        )
        main_node = HeroNode(
            id="test.py.main",
            type=NodeType.FUNCTION,
            name="main",
            content="def main(): helper(); process()"
        )
        helper_node = HeroNode(
            id="test.py.helper",
            type=NodeType.FUNCTION,
            name="helper",
            content="def helper(): process()"
        )
        process_node = HeroNode(
            id="test.py.process",
            type=NodeType.FUNCTION,
            name="process",
            content="def process(): pass"
        )
        
        # Add nodes
        self.graph.add_node(file_node)
        self.graph.add_node(main_node)
        self.graph.add_node(helper_node)
        self.graph.add_node(process_node)
        
        # Add edges
        self.graph.add_edge(file_node.id, main_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(file_node.id, helper_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(file_node.id, process_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(main_node.id, helper_node.id, EdgeType.CALLS)
        self.graph.add_edge(main_node.id, process_node.id, EdgeType.CALLS)
        self.graph.add_edge(helper_node.id, process_node.id, EdgeType.CALLS)
        
        # Track dependencies
        self.traverser._build_weighted_graph()
        
        # Find related locations for main function
        main_id = "test.py.main"
        related = self.traverser.find_related_locations([main_id])
        
        # Verify related locations
        self.assertEqual(len(related), 2)
        related_names = {node.name for node in related}
        self.assertEqual(related_names, {"helper", "process"})

class TestDependencyTracker(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.graph = HeroGraph()
        self.tracker = DependencyTracker(self.graph)
        
    def test_track_dependencies(self):
        """Test tracking code dependencies."""
        # Create test code with dependencies
        test_code = """
def main():
    helper()
"""
        # Add code to graph
        file_node = HeroNode(
            id="test.py",
            type=NodeType.FILE,
            name="test.py",
            content=test_code
        )
        main_node = HeroNode(
            id="test.py.main",
            type=NodeType.FUNCTION,
            name="main",
            content="def main(): helper()"
        )
        helper_node = HeroNode(
            id="test.py.helper",
            type=NodeType.FUNCTION,
            name="helper",
            content="def helper(): pass"
        )
        
        self.graph.add_node(file_node)
        self.graph.add_node(main_node)
        self.graph.add_node(helper_node)
        
        # Add edges
        self.graph.add_edge(file_node.id, main_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(file_node.id, helper_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(main_node.id, helper_node.id, EdgeType.CALLS)
        
        # Track dependencies
        self.tracker._build_dependency_graph()
        
        # Get dependencies for main function
        dependencies = self.tracker.get_dependencies("test.py.main")
        
        # Verify dependencies
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0].dependency_type, EdgeType.CALLS.value)
        
    def test_get_dependency_path(self):
        """Test finding dependency paths between nodes."""
        # Create test code with dependency chain
        test_code = """
def main():
    helper()
    
def helper():
    print("helping")
"""
        # Add code to graph
        file_node = HeroNode(
            id="test.py",
            type=NodeType.FILE,
            name="test.py",
            content=test_code
        )
        main_node = HeroNode(
            id="test.py.main",
            type=NodeType.FUNCTION,
            name="main",
            content="def main(): helper()"
        )
        helper_node = HeroNode(
            id="test.py.helper",
            type=NodeType.FUNCTION,
            name="helper",
            content="def helper(): print('helping')"
        )
        
        self.graph.add_node(file_node)
        self.graph.add_node(main_node)
        self.graph.add_node(helper_node)
        
        # Add edges
        self.graph.add_edge(file_node.id, main_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(file_node.id, helper_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(main_node.id, helper_node.id, EdgeType.CALLS)
        
        # Track dependencies
        self.tracker._build_dependency_graph()
        
        # Find dependency path
        path = self.tracker.get_dependency_path("test.py.main", "test.py.helper")
        
        # Verify path
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0].dependency_type, EdgeType.CALLS.value)

if __name__ == '__main__':
    unittest.main() 