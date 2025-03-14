from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import Action, MessageAction, AgentFinishAction
from openhands.llm.llm import LLM

from .graph_encoder.hero_graph import HeroGraph, HeroNode
from .reasoning import (
    LLMReasoner,
    ReasoningContext,
    GraphTraverser,
    TraversalConfig,
    DependencyTracker,
    DependencyInfo
)

@dataclass
class LocalizationResult:
    """Result from code localization."""
    node: HeroNode
    confidence: float
    reasoning: str
    context: Optional[Dict[str, Any]] = None

class LocAgent(Agent):
    """Agent specialized in code localization using graph-based reasoning."""
    
    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        """Initialize the LocAgent.
        
        Args:
            llm (LLM): The LLM to use for reasoning
            config (AgentConfig): Configuration for the agent
        """
        super().__init__(llm, config)
        
        # Initialize components
        self.graph = HeroGraph()
        self.llm_reasoner = LLMReasoner(self)
        self.graph_traverser = GraphTraverser(self.graph)
        self.dependency_tracker = DependencyTracker(self.graph)
        
        # Initialize state
        self.reset()
        
    def reset(self) -> None:
        """Reset the agent's state."""
        super().reset()
        self.graph = HeroGraph()
        self.graph_traverser = GraphTraverser(self.graph)
        self.dependency_tracker = DependencyTracker(self.graph)
        
    def step(self, state: State) -> Action:
        """Perform one step of code localization.
        
        Args:
            state (State): Current state containing the query and context
            
        Returns:
            Action: Next action to take
        """
        # Get the latest user message
        latest_message = state.get_last_user_message()
        if not latest_message:
            return AgentFinishAction()
            
        # Extract query from message
        query = latest_message.content
        
        # Find initial locations
        initial_locations = self._find_initial_locations(query)
        
        # Find related locations through graph traversal
        related_locations = self.graph_traverser.find_related_locations(initial_locations)
        
        # Create reasoning context
        context = ReasoningContext(
            query=query,
            initial_locations=initial_locations,
            related_locations=related_locations,
            graph_context=self.graph
        )
        
        # Perform LLM-based reasoning
        results = self.llm_reasoner.reason(context)
        
        # Process results
        if not results:
            return MessageAction(
                content="I couldn't find any relevant code locations for your query. "
                       "Could you please provide more details or clarify your request?"
            )
            
        # Sort results by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Format response
        response = self._format_localization_results(results)
        return MessageAction(content=response)
        
    def _find_initial_locations(self, query: str) -> List[HeroNode]:
        """Find initial code locations based on the query.
        
        Args:
            query (str): The user's query
            
        Returns:
            List[HeroNode]: List of initial code locations
        """
        # Extract keywords from query
        keywords = query.lower().split()
        
        # Search for nodes matching keywords
        matching_nodes = []
        for node in self.graph.graph.nodes(data=True):
            node_data = node[1]
            
            # Check node name
            if any(keyword in node_data['name'].lower() for keyword in keywords):
                matching_nodes.append(self.graph.get_node(node[0]))
                
            # Check node content
            if node_data.get('content') and any(keyword in node_data['content'].lower() for keyword in keywords):
                matching_nodes.append(self.graph.get_node(node[0]))
                
        return matching_nodes
        
    def _format_localization_results(self, results: List[Dict[str, Any]]) -> str:
        """Format localization results into a readable response.
        
        Args:
            results (List[Dict[str, Any]]): List of localization results
            
        Returns:
            str: Formatted response
        """
        response_parts = ["I found the following relevant code locations:"]
        
        for result in results[:5]:  # Show top 5 results
            node = self.graph.get_node(result['node_id'])
            if not node:
                continue
                
            response_parts.append(f"\n{node.name} ({node.type.value})")
            response_parts.append(f"Confidence: {result['confidence']:.2f}")
            response_parts.append(f"Reasoning: {result['reasoning']}")
            
            # Add context if available
            if node.context:
                response_parts.append(f"Context: {node.context}")
                
        return "\n".join(response_parts)
        
    def add_code_to_graph(self, code: str, file_path: str) -> None:
        """Add code to the graph representation.
        
        Args:
            code (str): The code to add
            file_path (str): Path to the source file
        """
        # Create file node
        file_node = HeroNode(
            id=file_path,
            type=NodeType.FILE,
            name=file_path,
            content=code
        )
        self.graph.add_node(file_node)
        
        # Parse code and add nodes/edges
        try:
            tree = ast.parse(code)
            self._process_ast(tree, file_node)
        except SyntaxError:
            logger.warning(f"Failed to parse code in {file_path}")
            
    def _process_ast(self, tree: ast.AST, parent_node: HeroNode) -> None:
        """Process AST nodes and add them to the graph.
        
        Args:
            tree (ast.AST): The AST to process
            parent_node (HeroNode): Parent node in the graph
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._add_function_node(node, parent_node)
            elif isinstance(node, ast.ClassDef):
                self._add_class_node(node, parent_node)
            elif isinstance(node, ast.Import):
                self._add_import_node(node, parent_node)
            elif isinstance(node, ast.ImportFrom):
                self._add_import_from_node(node, parent_node)
                
    def _add_function_node(self, node: ast.FunctionDef, parent_node: HeroNode) -> None:
        """Add a function node to the graph.
        
        Args:
            node (ast.FunctionDef): The function AST node
            parent_node (HeroNode): Parent node in the graph
        """
        func_node = HeroNode(
            id=f"{parent_node.id}.{node.name}",
            type=NodeType.FUNCTION,
            name=node.name,
            content=ast.unparse(node),
            start_line=node.lineno,
            end_line=node.end_lineno
        )
        self.graph.add_node(func_node)
        self.graph.add_edge(parent_node.id, func_node.id, EdgeType.CONTAINS)
        
    def _add_class_node(self, node: ast.ClassDef, parent_node: HeroNode) -> None:
        """Add a class node to the graph.
        
        Args:
            node (ast.ClassDef): The class AST node
            parent_node (HeroNode): Parent node in the graph
        """
        class_node = HeroNode(
            id=f"{parent_node.id}.{node.name}",
            type=NodeType.CLASS,
            name=node.name,
            content=ast.unparse(node),
            start_line=node.lineno,
            end_line=node.end_lineno
        )
        self.graph.add_node(class_node)
        self.graph.add_edge(parent_node.id, class_node.id, EdgeType.CONTAINS)
        
        # Process class body
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._add_function_node(item, class_node)
                
    def _add_import_node(self, node: ast.Import, parent_node: HeroNode) -> None:
        """Add an import node to the graph.
        
        Args:
            node (ast.Import): The import AST node
            parent_node (HeroNode): Parent node in the graph
        """
        for name in node.names:
            import_node = HeroNode(
                id=f"{parent_node.id}.import.{name.name}",
                type=NodeType.IMPORT,
                name=name.name,
                content=ast.unparse(node),
                start_line=node.lineno,
                end_line=node.end_lineno
            )
            self.graph.add_node(import_node)
            self.graph.add_edge(parent_node.id, import_node.id, EdgeType.IMPORTS)
            
    def _add_import_from_node(self, node: ast.ImportFrom, parent_node: HeroNode) -> None:
        """Add an import from node to the graph.
        
        Args:
            node (ast.ImportFrom): The import from AST node
            parent_node (HeroNode): Parent node in the graph
        """
        module_name = node.module or ''
        for name in node.names:
            import_node = HeroNode(
                id=f"{parent_node.id}.import.{module_name}.{name.name}",
                type=NodeType.IMPORT,
                name=f"{module_name}.{name.name}",
                content=ast.unparse(node),
                start_line=node.lineno,
                end_line=node.end_lineno
            )
            self.graph.add_node(import_node)
            self.graph.add_edge(parent_node.id, import_node.id, EdgeType.IMPORTS) 