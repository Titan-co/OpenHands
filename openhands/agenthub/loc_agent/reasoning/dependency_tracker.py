from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
import ast
import networkx as nx

from ..graph_encoder.hero_graph import HeroGraph, HeroNode, NodeType, EdgeType

@dataclass
class DependencyInfo:
    """Information about code dependencies."""
    source_id: str
    target_id: str
    dependency_type: str
    line_number: int
    context: Dict[str, Any]
    confidence: float

class DependencyTracker:
    """Advanced dependency tracking for code analysis."""
    
    def __init__(self, graph: HeroGraph):
        self.graph = graph
        self.dependency_graph = nx.DiGraph()
        self._build_dependency_graph()
        
    def _build_dependency_graph(self):
        """Build a graph of code dependencies."""
        # Initialize dependency graph
        self.dependency_graph = nx.DiGraph()
        
        # Track dependencies for each node
        for node in self.graph.graph.nodes(data=True):
            node_id = node[0]
            node_data = node[1]
            
            self._track_dependencies(node_id, node_data)
                
    def _track_dependencies(self, node_id: str, node_data: Dict[str, Any]):
        """Track dependencies for a node."""
        try:
            # Add node to dependency graph if not already present
            if node_id not in self.dependency_graph:
                self.dependency_graph.add_node(node_id, **node_data)
            
            # Track dependencies based on node type
            node_type = NodeType(node_data['type'])
            if node_type == NodeType.FUNCTION:
                self._track_function_dependencies(node_id, node_data)
            elif node_type == NodeType.CLASS:
                self._track_class_dependencies(node_id, node_data)
            elif node_type == NodeType.FILE:
                self._track_file_dependencies(node_id, node_data)
            elif node_type == NodeType.METHOD:
                self._track_method_dependencies(node_id, node_data)
                
        except Exception:
            # Handle tracking errors
            pass
            
    def _track_function_dependencies(self, node_id: str, node_data: Dict[str, Any]):
        """Track dependencies for a function node."""
        try:
            # Add node to dependency graph if not already present
            if node_id not in self.dependency_graph:
                self.dependency_graph.add_node(node_id, **node_data)
            
            # Parse function body
            tree = ast.parse(node_data['content'])
            
            # Track function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    self._track_function_call(node_id, node, node_data)
                    
            # Track variable references
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    self._track_variable_reference(node_id, node, node_data)
                    
        except SyntaxError:
            # Handle invalid syntax
            pass
            
    def _track_class_dependencies(self, node_id: str, node_data: Dict[str, Any]):
        """Track dependencies for a class node."""
        try:
            # Add node to dependency graph if not already present
            if node_id not in self.dependency_graph:
                self.dependency_graph.add_node(node_id, **node_data)
            
            # Parse class body
            tree = ast.parse(node_data['content'])
            
            # Track base classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        self._track_class_inheritance(node_id, base, node_data)
                        
            # Track method dependencies
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._track_method_dependencies(node_id, node, node_data)
                    
        except SyntaxError:
            # Handle invalid syntax
            pass
            
    def _track_file_dependencies(self, node_id: str, node_data: Dict[str, Any]):
        """Track dependencies for a file node."""
        try:
            # Add node to dependency graph if not already present
            if node_id not in self.dependency_graph:
                self.dependency_graph.add_node(node_id, **node_data)
            
            # Parse file content
            tree = ast.parse(node_data['content'])
            
            # Track imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._track_import(node_id, node, node_data)
                    
        except SyntaxError:
            # Handle invalid syntax
            pass
            
    def _track_function_call(self, caller_id: str, call_node: ast.Call, context: Dict[str, Any]):
        """Track a function call dependency."""
        try:
            # Get function name
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id
            elif isinstance(call_node.func, ast.Attribute):
                func_name = call_node.func.attr
            else:
                return
                
            # Find called function
            called_func = self._find_function(func_name)
            if called_func:
                # Add dependency edge
                self.dependency_graph.add_edge(
                    caller_id,
                    called_func.id,
                    edge_type=EdgeType.CALLS.value,
                    line_number=call_node.lineno,
                    context={
                        'call_args': self._get_call_args(call_node),
                        'caller_context': context
                    }
                )
                
        except Exception:
            # Handle tracking errors
            pass
            
    def _track_variable_reference(self, node_id: str, name_node: ast.Name, context: Dict[str, Any]):
        """Track a variable reference dependency."""
        try:
            # Find variable definition
            var_def = self._find_variable(name_node.id)
            if var_def:
                # Add dependency edge
                self.dependency_graph.add_edge(
                    node_id,
                    var_def.id,
                    edge_type=EdgeType.REFERENCES.value,
                    line_number=name_node.lineno,
                    context={
                        'variable_name': name_node.id,
                        'reference_context': context
                    }
                )
                
        except Exception:
            # Handle tracking errors
            pass
            
    def _track_class_inheritance(self, class_id: str, base_node: ast.expr, context: Dict[str, Any]):
        """Track a class inheritance dependency."""
        try:
            # Get base class name
            if isinstance(base_node, ast.Name):
                base_name = base_node.id
            elif isinstance(base_node, ast.Attribute):
                base_name = base_node.attr
            else:
                return
                
            # Find base class
            base_class = self._find_class(base_name)
            if base_class:
                # Add dependency edge
                self.dependency_graph.add_edge(
                    class_id,
                    base_class.id,
                    edge_type=EdgeType.INHERITS.value,
                    line_number=base_node.lineno,
                    context={
                        'base_class_name': base_name,
                        'inheritance_context': context
                    }
                )
                
        except Exception:
            # Handle tracking errors
            pass
            
    def _track_method_dependencies(self, class_id: str, method_node: ast.FunctionDef, context: Dict[str, Any]):
        """Track dependencies for a class method."""
        try:
            # Create method node
            method_id = f"{class_id}.{method_node.name}"
            
            # Track method dependencies
            for node in ast.walk(method_node):
                if isinstance(node, ast.Call):
                    self._track_function_call(method_id, node, context)
                elif isinstance(node, ast.Name):
                    self._track_variable_reference(method_id, node, context)
                    
        except Exception:
            # Handle tracking errors
            pass
            
    def _track_import(self, file_id: str, import_node: ast.AST, context: Dict[str, Any]):
        """Track an import dependency."""
        try:
            # Add file node to dependency graph if not already present
            if file_id not in self.dependency_graph:
                self.dependency_graph.add_node(file_id)
            
            if isinstance(import_node, ast.Import):
                for name in import_node.names:
                    imported_module = self._find_module(name.name)
                    if imported_module:
                        # Add imported module node if not present
                        if imported_module.id not in self.dependency_graph:
                            self.dependency_graph.add_node(imported_module.id)
                        self.dependency_graph.add_edge(
                            file_id,
                            imported_module.id,
                            edge_type=EdgeType.IMPORTS.value,
                            line_number=import_node.lineno,
                            context={
                                'import_name': name.name,
                                'import_alias': name.asname,
                                'import_context': context
                            }
                        )
            elif isinstance(import_node, ast.ImportFrom):
                module_name = import_node.module or ''
                for name in import_node.names:
                    imported_name = f"{module_name}.{name.name}"
                    imported_item = self._find_imported_item(imported_name)
                    if imported_item:
                        # Add imported item node if not present
                        if imported_item.id not in self.dependency_graph:
                            self.dependency_graph.add_node(imported_item.id)
                        self.dependency_graph.add_edge(
                            file_id,
                            imported_item.id,
                            edge_type=EdgeType.IMPORTS.value,
                            line_number=import_node.lineno,
                            context={
                                'import_name': imported_name,
                                'import_alias': name.asname,
                                'import_context': context
                            }
                        )
                        
        except Exception:
            # Handle tracking errors
            pass
            
    def _find_function(self, name: str) -> Optional[HeroNode]:
        """Find a function node by name."""
        for node in self.graph.graph.nodes(data=True):
            if (node[1]['type'] == NodeType.FUNCTION and 
                node[1]['name'] == name):
                return self.graph.get_node(node[0])
        return None
        
    def _find_variable(self, name: str) -> Optional[HeroNode]:
        """Find a variable node by name."""
        for node in self.graph.graph.nodes(data=True):
            if (node[1]['type'] == NodeType.VARIABLE and 
                node[1]['name'] == name):
                return self.graph.get_node(node[0])
        return None
        
    def _find_class(self, name: str) -> Optional[HeroNode]:
        """Find a class node by name."""
        for node in self.graph.graph.nodes(data=True):
            if (node[1]['type'] == NodeType.CLASS and 
                node[1]['name'] == name):
                return self.graph.get_node(node[0])
        return None
        
    def _find_module(self, name: str) -> Optional[HeroNode]:
        """Find a module node by name."""
        for node in self.graph.graph.nodes(data=True):
            if (node[1]['type'] == NodeType.FILE and 
                node[1]['name'] == name):
                return self.graph.get_node(node[0])
        return None
        
    def _find_imported_item(self, name: str) -> Optional[HeroNode]:
        """Find an imported item by name."""
        for node in self.graph.graph.nodes(data=True):
            if node[1]['name'] == name:
                return self.graph.get_node(node[0])
        return None
        
    def _get_call_args(self, call_node: ast.Call) -> List[Dict[str, Any]]:
        """Get information about function call arguments."""
        args = []
        for arg in call_node.args:
            if isinstance(arg, ast.Name):
                args.append({
                    'type': 'name',
                    'value': arg.id
                })
            elif isinstance(arg, ast.Constant):
                args.append({
                    'type': 'constant',
                    'value': arg.value
                })
            else:
                args.append({
                    'type': 'other',
                    'value': ast.unparse(arg)
                })
        return args
        
    def get_dependencies(self, node_id: str) -> List[DependencyInfo]:
        """Get all dependencies for a node."""
        dependencies = []
        for _, target_id, edge_data in self.dependency_graph.edges(node_id, data=True):
            dependencies.append(DependencyInfo(
                source_id=node_id,
                target_id=target_id,
                dependency_type=edge_data['edge_type'],
                line_number=edge_data['line_number'],
                context=edge_data['context'],
                confidence=self._calculate_dependency_confidence(edge_data)
            ))
        return dependencies
        
    def _calculate_dependency_confidence(self, edge_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a dependency."""
        # Base confidence on dependency type
        type_confidence = {
            'CALLS': 0.9,
            'REFERENCES': 0.8,
            'INHERITS': 0.7,
            'IMPORTS': 0.6
        }.get(edge_data['edge_type'], 0.5)
        
        # Adjust confidence based on context
        context = edge_data['context']
        if 'call_args' in context:
            # Function call confidence
            return type_confidence * 1.1
        elif 'variable_name' in context:
            # Variable reference confidence
            return type_confidence * 1.0
        elif 'base_class_name' in context:
            # Class inheritance confidence
            return type_confidence * 1.2
        elif 'import_name' in context:
            # Import confidence
            return type_confidence * 0.9
            
        return type_confidence
        
    def get_dependency_path(self, source_id: str, target_id: str) -> List[DependencyInfo]:
        """Get the dependency path between two nodes."""
        try:
            # Find shortest path
            path = nx.shortest_path(self.dependency_graph, source_id, target_id)
            
            # Get dependency information for each edge
            dependencies = []
            for i in range(len(path) - 1):
                edge_data = self.dependency_graph.edges[path[i], path[i + 1]]
                dependencies.append(DependencyInfo(
                    source_id=path[i],
                    target_id=path[i + 1],
                    dependency_type=edge_data['edge_type'],
                    line_number=edge_data['line_number'],
                    context=edge_data['context'],
                    confidence=self._calculate_dependency_confidence(edge_data)
                ))
                
            return dependencies
            
        except nx.NetworkXNoPath:
            return []
            
    def get_dependency_subgraph(self, node_ids: Set[str]) -> nx.DiGraph:
        """Get a subgraph of dependencies for specified nodes."""
        return self.dependency_graph.subgraph(node_ids) 