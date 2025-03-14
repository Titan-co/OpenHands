from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from networkx.classes.digraph import DiGraph

class NodeType(Enum):
    """Types of nodes in the HERO graph."""
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"
    IMPORT = "import"
    METHOD = "method"
    PARAMETER = "parameter"
    RETURN = "return"
    DECORATOR = "decorator"

class EdgeType(Enum):
    """Types of edges in the HERO graph."""
    CONTAINS = "contains"
    CALLS = "calls"
    REFERENCES = "references"
    INHERITS = "inherits"
    IMPORTS = "imports"
    RETURNS = "returns"
    DECORATES = "decorates"
    PARAMETER_OF = "parameter_of"

@dataclass
class HeroNode:
    """Represents a node in the HERO graph with rich metadata."""
    id: str
    type: NodeType
    name: str
    content: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    context: Optional[str] = None

class HeroGraph:
    """Hierarchical Entity-Relation Ontology graph for code representation."""
    
    def __init__(self):
        self.graph = DiGraph()
        self.node_types: Dict[NodeType, Set[str]] = {
            node_type: set() for node_type in NodeType
        }
        self.edge_types: Dict[EdgeType, Set[tuple]] = {
            edge_type: set() for edge_type in EdgeType
        }
        
    def add_node(self, node: HeroNode) -> None:
        """Add a node to the graph with its type."""
        self.graph.add_node(node.id, **node.__dict__)
        self.node_types[node.type].add(node.id)
        
    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an edge to the graph with its type and metadata."""
        self.graph.add_edge(source_id, target_id, type=edge_type.value, **metadata or {})
        self.edge_types[edge_type].add((source_id, target_id))
        
    def get_node(self, node_id: str) -> Optional[HeroNode]:
        """Get a node by its ID."""
        if node_id in self.graph:
            data = self.graph.nodes[node_id]
            return HeroNode(**data)
        return None
        
    def get_nodes_by_type(self, node_type: NodeType) -> List[HeroNode]:
        """Get all nodes of a specific type."""
        return [self.get_node(node_id) for node_id in self.node_types[node_type]]
        
    def get_edges_by_type(self, edge_type: EdgeType) -> List[tuple]:
        """Get all edges of a specific type."""
        return list(self.edge_types[edge_type])
        
    def get_children(self, node_id: str) -> List[HeroNode]:
        """Get all child nodes of a given node."""
        return [self.get_node(child_id) for child_id in self.graph.successors(node_id)]
        
    def get_parents(self, node_id: str) -> List[HeroNode]:
        """Get all parent nodes of a given node."""
        return [self.get_node(parent_id) for parent_id in self.graph.predecessors(node_id)]
        
    def get_edges(self, node_id: str) -> List[tuple]:
        """Get all edges connected to a node."""
        return list(self.graph.edges(node_id, data=True))
        
    def get_paths(self, source_id: str, target_id: str, max_hops: int = 3) -> List[List[str]]:
        """Get all paths between two nodes up to a maximum number of hops."""
        try:
            return list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_hops))
        except nx.NetworkXNoPath:
            return []
            
    def get_subgraph(self, node_ids: Set[str]) -> 'HeroGraph':
        """Get a subgraph containing only the specified nodes and their edges."""
        subgraph = HeroGraph()
        subgraph.graph = self.graph.subgraph(node_ids).copy()
        
        # Update node types
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                subgraph.node_types[node.type].add(node_id)
                
        # Update edge types
        for edge in subgraph.graph.edges(data=True):
            edge_type = EdgeType(edge[2]['type'])
            subgraph.edge_types[edge_type].add((edge[0], edge[1]))
            
        return subgraph
        
    def get_context_subgraph(self, node_id: str, max_hops: int = 2) -> 'HeroGraph':
        """Get a subgraph containing the context around a node."""
        # Get all nodes within max_hops distance
        context_nodes = set()
        for hop in range(max_hops + 1):
            for path in nx.single_source_shortest_path_length(self.graph, node_id, cutoff=hop):
                context_nodes.add(path)
                
        return self.get_subgraph(context_nodes)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the graph to a dictionary representation."""
        return {
            'nodes': {
                node_type.value: [self.get_node(node_id).__dict__ 
                                for node_id in node_ids]
                for node_type, node_ids in self.node_types.items()
            },
            'edges': {
                edge_type.value: [(source, target, self.graph.edges[source, target])
                                for source, target in edge_pairs]
                for edge_type, edge_pairs in self.edge_types.items()
            }
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeroGraph':
        """Create a graph from a dictionary representation."""
        graph = cls()
        
        # Add nodes
        for node_type, nodes in data['nodes'].items():
            for node_data in nodes:
                node = HeroNode(**node_data)
                graph.add_node(node)
                
        # Add edges
        for edge_type, edges in data['edges'].items():
            for source, target, edge_data in edges:
                graph.add_edge(source, target, EdgeType(edge_type), edge_data)
                
        return graph 