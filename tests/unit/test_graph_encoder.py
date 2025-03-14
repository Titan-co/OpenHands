import unittest
from openhands.agenthub.loc_agent.graph_encoder.hero_graph import (
    HeroGraph,
    HeroNode,
    NodeType,
    EdgeType
)

class TestHeroGraph(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.graph = HeroGraph()
        
    def test_add_node(self):
        """Test adding nodes to the graph."""
        node = HeroNode(
            id="test",
            type=NodeType.FILE,
            name="test.py",
            content="def test(): pass"
        )
        
        self.graph.add_node(node)
        self.assertTrue(self.graph.has_node("test"))
        self.assertEqual(self.graph.get_node("test"), node)
        
    def test_add_edge(self):
        """Test adding edges to the graph."""
        # Create nodes
        file_node = HeroNode(
            id="test.py",
            type=NodeType.FILE,
            name="test.py",
            content="def test(): pass"
        )
        func_node = HeroNode(
            id="test.py.test",
            type=NodeType.FUNCTION,
            name="test",
            content="def test(): pass"
        )
        
        # Add nodes and edge
        self.graph.add_node(file_node)
        self.graph.add_node(func_node)
        self.graph.add_edge(file_node.id, func_node.id, EdgeType.CONTAINS)
        
        # Verify edge
        self.assertTrue(self.graph.has_edge(file_node.id, func_node.id))
        edge_data = self.graph.graph.get_edge_data(file_node.id, func_node.id)
        self.assertEqual(edge_data['edge_type'], EdgeType.CONTAINS.value)
        
    def test_get_nodes_by_type(self):
        """Test retrieving nodes by type."""
        # Add nodes of different types
        file_node = HeroNode(
            id="test.py",
            type=NodeType.FILE,
            name="test.py",
            content="def test(): pass"
        )
        func_node = HeroNode(
            id="test.py.test",
            type=NodeType.FUNCTION,
            name="test",
            content="def test(): pass"
        )
        class_node = HeroNode(
            id="test.py.TestClass",
            type=NodeType.CLASS,
            name="TestClass",
            content="class TestClass: pass"
        )
        
        self.graph.add_node(file_node)
        self.graph.add_node(func_node)
        self.graph.add_node(class_node)
        
        # Test getting nodes by type
        func_nodes = self.graph.get_nodes_by_type(NodeType.FUNCTION)
        self.assertEqual(len(func_nodes), 1)
        self.assertEqual(func_nodes[0], func_node)
        
        class_nodes = self.graph.get_nodes_by_type(NodeType.CLASS)
        self.assertEqual(len(class_nodes), 1)
        self.assertEqual(class_nodes[0], class_node)
        
    def test_get_edges_by_type(self):
        """Test retrieving edges by type."""
        # Create nodes and edges
        file_node = HeroNode(
            id="test.py",
            type=NodeType.FILE,
            name="test.py",
            content="def test(): pass"
        )
        func_node = HeroNode(
            id="test.py.test",
            type=NodeType.FUNCTION,
            name="test",
            content="def test(): pass"
        )
        
        self.graph.add_node(file_node)
        self.graph.add_node(func_node)
        self.graph.add_edge(file_node.id, func_node.id, EdgeType.CONTAINS)
        
        # Test getting edges by type
        contains_edges = self.graph.get_edges_by_type(EdgeType.CONTAINS)
        self.assertEqual(len(contains_edges), 1)
        self.assertEqual(contains_edges[0][0], file_node.id)
        self.assertEqual(contains_edges[0][1], func_node.id)
        
    def test_get_subgraph(self):
        """Test getting a subgraph containing specific nodes."""
        # Create a larger graph
        file_node = HeroNode(
            id="test.py",
            type=NodeType.FILE,
            name="test.py",
            content="def test(): pass"
        )
        func1_node = HeroNode(
            id="test.py.test1",
            type=NodeType.FUNCTION,
            name="test1",
            content="def test1(): pass"
        )
        func2_node = HeroNode(
            id="test.py.test2",
            type=NodeType.FUNCTION,
            name="test2",
            content="def test2(): pass"
        )
        
        self.graph.add_node(file_node)
        self.graph.add_node(func1_node)
        self.graph.add_node(func2_node)
        self.graph.add_edge(file_node.id, func1_node.id, EdgeType.CONTAINS)
        self.graph.add_edge(file_node.id, func2_node.id, EdgeType.CONTAINS)
        
        # Get subgraph with file and first function
        subgraph = self.graph.get_subgraph([file_node.id, func1_node.id])
        
        # Verify subgraph
        self.assertEqual(len(subgraph.nodes), 2)
        self.assertEqual(len(subgraph.edges), 1)
        self.assertTrue(subgraph.has_node(file_node.id))
        self.assertTrue(subgraph.has_node(func1_node.id))
        
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization of the graph."""
        # Create a graph with nodes and edges
        file_node = HeroNode(
            id="test.py",
            type=NodeType.FILE,
            name="test.py",
            content="def test(): pass"
        )
        func_node = HeroNode(
            id="test.py.test",
            type=NodeType.FUNCTION,
            name="test",
            content="def test(): pass"
        )
        
        self.graph.add_node(file_node)
        self.graph.add_node(func_node)
        self.graph.add_edge(file_node.id, func_node.id, EdgeType.CONTAINS)
        
        # Convert to dict
        graph_dict = self.graph.to_dict()
        
        # Create new graph from dict
        new_graph = HeroGraph.from_dict(graph_dict)
        
        # Verify new graph matches original
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(new_graph.edges), len(self.graph.edges))
        self.assertTrue(new_graph.has_node(file_node.id))
        self.assertTrue(new_graph.has_node(func_node.id))
        self.assertTrue(new_graph.has_edge(file_node.id, func_node.id))

if __name__ == '__main__':
    unittest.main() 