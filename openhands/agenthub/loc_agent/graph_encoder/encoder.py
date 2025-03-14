from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import ast
import networkx as nx
from networkx.classes.digraph import DiGraph

@dataclass
class CodeNode:
    """Represents a node in the code graph."""
    id: str
    type: str  # 'file', 'class', 'function', 'import'
    name: str
    content: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class CodeGraphEncoder:
    """Encodes codebase into a graph representation for code localization."""
    
    def __init__(self):
        self.graph = DiGraph()
        
    def encode_codebase(self, workspace_path: str) -> DiGraph:
        """Encode the entire codebase into a graph representation.
        
        Args:
            workspace_path: Path to the workspace root
            
        Returns:
            DiGraph: NetworkX graph representing the codebase structure
        """
        # Reset graph
        self.graph.clear()
        
        # Scan all Python files
        for file_path in Path(workspace_path).rglob("*.py"):
            self._process_file(file_path, workspace_path)
            
        return self.graph
        
    def _process_file(self, file_path: Path, workspace_path: str):
        """Process a single Python file and add its structure to the graph."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Create file node
            relative_path = str(file_path.relative_to(workspace_path))
            file_node = CodeNode(
                id=f"file_{relative_path}",
                type="file",
                name=relative_path,
                content=content
            )
            self.graph.add_node(file_node.id, **file_node.__dict__)
            
            # Parse AST and extract structure
            tree = ast.parse(content)
            self._process_ast(tree, file_node.id)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    def _process_ast(self, tree: ast.AST, parent_id: str):
        """Process AST nodes and add them to the graph."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._process_import(node, parent_id)
            elif isinstance(node, ast.ImportFrom):
                self._process_import_from(node, parent_id)
            elif isinstance(node, ast.ClassDef):
                self._process_class(node, parent_id)
            elif isinstance(node, ast.FunctionDef):
                self._process_function(node, parent_id)
                
    def _process_import(self, node: ast.Import, parent_id: str):
        """Process import statements."""
        for name in node.names:
            import_node = CodeNode(
                id=f"import_{name.name}",
                type="import",
                name=name.name,
                start_line=node.lineno,
                end_line=node.end_lineno
            )
            self.graph.add_node(import_node.id, **import_node.__dict__)
            self.graph.add_edge(parent_id, import_node.id, type="imports")
            
    def _process_import_from(self, node: ast.ImportFrom, parent_id: str):
        """Process from-import statements."""
        module = node.module or ""
        for name in node.names:
            import_node = CodeNode(
                id=f"import_{module}.{name.name}",
                type="import",
                name=f"{module}.{name.name}",
                start_line=node.lineno,
                end_line=node.end_lineno
            )
            self.graph.add_node(import_node.id, **import_node.__dict__)
            self.graph.add_edge(parent_id, import_node.id, type="imports")
            
    def _process_class(self, node: ast.ClassDef, parent_id: str):
        """Process class definitions."""
        class_node = CodeNode(
            id=f"class_{node.name}",
            type="class",
            name=node.name,
            start_line=node.lineno,
            end_line=node.end_lineno,
            metadata={
                "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                "decorators": [decorator.id for decorator in node.decorator_list if isinstance(decorator, ast.Name)]
            }
        )
        self.graph.add_node(class_node.id, **class_node.__dict__)
        self.graph.add_edge(parent_id, class_node.id, type="contains")
        
        # Process class body
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._process_function(item, class_node.id)
                
    def _process_function(self, node: ast.FunctionDef, parent_id: str):
        """Process function definitions."""
        function_node = CodeNode(
            id=f"function_{node.name}",
            type="function",
            name=node.name,
            start_line=node.lineno,
            end_line=node.end_lineno,
            metadata={
                "args": [arg.arg for arg in node.args.args],
                "decorators": [decorator.id for decorator in node.decorator_list if isinstance(decorator, ast.Name)]
            }
        )
        self.graph.add_node(function_node.id, **function_node.__dict__)
        self.graph.add_edge(parent_id, function_node.id, type="contains")
        
    def get_node_by_id(self, node_id: str) -> Optional[CodeNode]:
        """Get a CodeNode by its ID."""
        if node_id in self.graph:
            data = self.graph.nodes[node_id]
            return CodeNode(**data)
        return None
        
    def get_children(self, node_id: str) -> List[CodeNode]:
        """Get all child nodes of a given node."""
        return [self.get_node_by_id(child_id) for child_id in self.graph.successors(node_id)]
        
    def get_parents(self, node_id: str) -> List[CodeNode]:
        """Get all parent nodes of a given node."""
        return [self.get_node_by_id(parent_id) for parent_id in self.graph.predecessors(node_id)] 