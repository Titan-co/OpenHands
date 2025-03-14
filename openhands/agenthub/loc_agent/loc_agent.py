import ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import Action, MessageAction, AgentFinishAction
from openhands.llm.llm import LLM

from .graph_encoder.hero_graph import HeroGraph, HeroNode, NodeType, EdgeType
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
        
        # Initialize logger
        self.logger = logger
        
        # Initialize components
        self.graph = HeroGraph()
        self.llm_reasoner = LLMReasoner(self)
        self.reasoner = self.llm_reasoner
        self.graph_traverser = GraphTraverser(self.graph)
        self.dependency_tracker = DependencyTracker(self.graph)
        
        # Initialize state
        from .mocks import State as MockState
        self.state = MockState()
        
        # Initialize state
        self.reset()
        
    def reset(self) -> None:
        """Reset the agent's state."""
        super().reset()
        self.graph = HeroGraph()
        self.graph_traverser = GraphTraverser(self.graph)
        self.dependency_tracker = DependencyTracker(self.graph)
        
        # Initialize state
        from .mocks import State as MockState
        self.state = MockState()
        
    def step(self, state: Optional[State] = None) -> MessageAction:
        """Take a step in the agent's execution."""
        # Use provided state or current state
        state = state or self.state
        
        # Get last user message
        last_message = state.get_last_user_message()
        if not last_message:
            return MessageAction(content="No message found in state.")
        
        # Find initial locations
        initial_locations = self._find_initial_locations(last_message.content)
        if not initial_locations:
            return MessageAction(content="I couldn't find any relevant code locations for your query. Could you please provide more details or clarify your request?")
        
        # Find related locations
        related_locations = self.graph_traverser.find_related_locations(initial_locations)
        
        # Create reasoning context
        context = ReasoningContext(
            query=last_message.content,
            initial_locations=initial_locations,
            related_locations=related_locations,
            graph_context=self.graph
        )
        
        # Get reasoning results
        results = self.llm_reasoner.reason(context)
        
        # Format results
        response = self._format_localization_results(results)
        
        return MessageAction(content=response)
        
    def _find_initial_locations(self, query: str) -> List[HeroNode]:
        """Find initial code locations based on query."""
        nodes = []
        seen_ids = set()
        
        # Search in function, class, and method nodes using individual query words
        query_terms = query.lower().split()
        for node_type in [NodeType.FUNCTION, NodeType.CLASS, NodeType.METHOD]:
            for node in self.graph.get_nodes_by_type(node_type):
                if node.id not in seen_ids:
                    if any(term in node.name.lower() for term in query_terms):
                        nodes.append(node)
                        seen_ids.add(node.id)
        
        return nodes
        
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
        # First pass: collect all function definitions
        function_nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_node = self._add_function_node(node, parent_node)
                function_nodes.append((func_node, node))
            elif isinstance(node, ast.ClassDef):
                self._add_class_node(node, parent_node)
            elif isinstance(node, ast.Import):
                self._add_import_node(node, parent_node)
            elif isinstance(node, ast.ImportFrom):
                self._add_import_from_node(node, parent_node)
        
        # Second pass: add CALLS edges for function calls
        for func_node, func_ast in function_nodes:
            for child in ast.walk(func_ast):
                if isinstance(child, ast.Call):
                    # Handle direct function calls (e.g., helper())
                    if isinstance(child.func, ast.Name):
                        called_name = child.func.id
                        target_id = f"{parent_node.id}.{called_name}"
                        if self.graph.has_node(target_id):
                            self.graph.add_edge(func_node.id, target_id, EdgeType.CALLS)
                    # Handle method calls (e.g., obj.method())
                    elif isinstance(child.func, ast.Attribute):
                        method_name = child.func.attr
                        if isinstance(child.func.value, ast.Name):
                            obj_name = child.func.value.id
                            target_id = f"{parent_node.id}.{obj_name}.{method_name}"
                            if self.graph.has_node(target_id):
                                self.graph.add_edge(func_node.id, target_id, EdgeType.CALLS)
        
    def _add_function_node(self, node: ast.FunctionDef, parent_node: HeroNode) -> HeroNode:
        """Add a function node to the graph.
        
        Args:
            node (ast.FunctionDef): The function AST node
            parent_node (HeroNode): Parent node in the graph
            
        Returns:
            HeroNode: The created function node
        """
        node_type = NodeType.METHOD if parent_node.type == NodeType.CLASS else NodeType.FUNCTION
        func_node = HeroNode(
            id=f"{parent_node.id}.{node.name}",
            type=node_type,
            name=node.name,
            content=ast.unparse(node),
            start_line=node.lineno,
            end_line=node.end_lineno
        )
        
        # Only add the node if it doesn't already exist
        if not self.graph.has_node(func_node.id):
            self.graph.add_node(func_node)
            self.graph.add_edge(parent_node.id, func_node.id, EdgeType.CONTAINS)
            
        return func_node
        
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