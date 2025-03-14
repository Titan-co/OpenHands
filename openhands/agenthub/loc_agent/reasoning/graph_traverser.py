from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
import networkx as nx

from ..graph_encoder.hero_graph import HeroGraph, HeroNode, NodeType, EdgeType

@dataclass
class TraversalConfig:
    """Configuration for graph traversal."""
    max_hops: int = 3
    min_confidence: float = 0.3
    edge_weights: Dict[EdgeType, float] = None
    node_type_weights: Dict[NodeType, float] = None
    
    def __post_init__(self):
        # Default edge weights
        if self.edge_weights is None:
            self.edge_weights = {
                EdgeType.CONTAINS: 1.0,
                EdgeType.CALLS: 0.8,
                EdgeType.REFERENCES: 0.6,
                EdgeType.INHERITS: 0.7,
                EdgeType.IMPORTS: 0.5,
                EdgeType.RETURNS: 0.4,
                EdgeType.DECORATES: 0.3,
                EdgeType.PARAMETER_OF: 0.2
            }
            
        # Default node type weights
        if self.node_type_weights is None:
            self.node_type_weights = {
                NodeType.FILE: 1.0,
                NodeType.CLASS: 0.9,
                NodeType.FUNCTION: 0.8,
                NodeType.METHOD: 0.7,
                NodeType.VARIABLE: 0.6,
                NodeType.IMPORT: 0.5,
                NodeType.PARAMETER: 0.4,
                NodeType.RETURN: 0.3,
                NodeType.DECORATOR: 0.2
            }

class GraphTraverser:
    """Graph traversal for multi-hop reasoning in code localization."""
    
    def __init__(self, graph: HeroGraph, config: Optional[TraversalConfig] = None):
        self.graph = graph
        self.config = config or TraversalConfig()
        self._build_weighted_graph()
        
    def _build_weighted_graph(self):
        """Build a weighted graph for traversal."""
        self.weighted_graph = nx.DiGraph()
        
        # Add nodes with weights
        for node in self.graph.graph.nodes(data=True):
            node_type = NodeType(node[1]['type'])
            weight = self.config.node_type_weights.get(node_type, 0.5)
            self.weighted_graph.add_node(node[0], weight=weight, **node[1])
            
        # Add edges with weights
        for edge in self.graph.graph.edges(data=True):
            edge_type = EdgeType(edge[2]['type'])
            weight = self.config.edge_weights.get(edge_type, 0.5)
            self.weighted_graph.add_edge(edge[0], edge[1], weight=weight, **edge[2])
            
    def find_related_locations(self, initial_nodes: List[HeroNode]) -> List[HeroNode]:
        """Find related locations through multi-hop traversal."""
        related_nodes = set()
        
        # Add initial nodes
        for node in initial_nodes:
            related_nodes.add(node.id)
            
        # Perform multi-hop traversal
        for hop in range(self.config.max_hops):
            new_nodes = set()
            
            # Traverse forward
            for node_id in related_nodes:
                for _, target_id, edge_data in self.weighted_graph.edges(node_id, data=True):
                    if target_id not in related_nodes:
                        # Calculate path weight
                        path_weight = self._calculate_path_weight(node_id, target_id)
                        if path_weight >= self.config.min_confidence:
                            new_nodes.add(target_id)
                            
            # Traverse backward
            for node_id in related_nodes:
                for source_id, _, edge_data in self.weighted_graph.edges(None, node_id, data=True):
                    if source_id not in related_nodes:
                        # Calculate path weight
                        path_weight = self._calculate_path_weight(source_id, node_id)
                        if path_weight >= self.config.min_confidence:
                            new_nodes.add(source_id)
                            
            related_nodes.update(new_nodes)
            
        # Convert node IDs to HeroNodes
        return [self.graph.get_node(node_id) for node_id in related_nodes]
        
    def _calculate_path_weight(self, source_id: str, target_id: str) -> float:
        """Calculate the weight of a path between two nodes."""
        try:
            # Find shortest path
            path = nx.shortest_path(self.weighted_graph, source_id, target_id)
            
            # Calculate path weight
            weight = 1.0
            for i in range(len(path) - 1):
                edge_data = self.weighted_graph.edges[path[i], path[i + 1]]
                weight *= edge_data['weight']
                
            return weight
            
        except nx.NetworkXNoPath:
            return 0.0
            
    def find_relevant_paths(self, source_id: str, target_id: str) -> List[List[str]]:
        """Find relevant paths between two nodes."""
        try:
            # Find all simple paths up to max_hops
            paths = list(nx.all_simple_paths(
                self.weighted_graph, 
                source_id, 
                target_id, 
                cutoff=self.config.max_hops
            ))
            
            # Calculate path weights and filter
            weighted_paths = []
            for path in paths:
                weight = self._calculate_path_weight(path[0], path[-1])
                if weight >= self.config.min_confidence:
                    weighted_paths.append((path, weight))
                    
            # Sort by weight
            weighted_paths.sort(key=lambda x: x[1], reverse=True)
            
            return [path for path, _ in weighted_paths]
            
        except nx.NetworkXNoPath:
            return []
            
    def get_node_context(self, node_id: str) -> Dict[str, Any]:
        """Get context information for a node."""
        context = {
            'node': self.graph.get_node(node_id),
            'parents': self.graph.get_parents(node_id),
            'children': self.graph.get_children(node_id),
            'incoming_edges': self.graph.get_edges(node_id),
            'outgoing_edges': self.graph.get_edges(node_id)
        }
        
        # Add path information to related nodes
        context['paths'] = {}
        for other_id in self.graph.graph.nodes():
            if other_id != node_id:
                paths = self.find_relevant_paths(node_id, other_id)
                if paths:
                    context['paths'][other_id] = paths
                    
        return context
        
    def get_subgraph_with_context(self, node_ids: Set[str]) -> HeroGraph:
        """Get a subgraph containing the specified nodes and their context."""
        # Get the subgraph
        subgraph = self.graph.get_subgraph(node_ids)
        
        # Add context nodes
        context_nodes = set()
        for node_id in node_ids:
            # Get nodes within max_hops distance
            for hop in range(self.config.max_hops + 1):
                for path in nx.single_source_shortest_path_length(
                    self.weighted_graph, 
                    node_id, 
                    cutoff=hop
                ):
                    context_nodes.add(path)
                    
        # Create context subgraph
        context_subgraph = self.graph.get_subgraph(context_nodes)
        
        return context_subgraph 