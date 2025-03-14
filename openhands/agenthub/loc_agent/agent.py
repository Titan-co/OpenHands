from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import ast

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.events.action import Action, FileReadAction, FileWriteAction, MessageAction
from openhands.events.observation import Observation

@dataclass
class CodeLocation:
    file_path: str
    start_line: int
    end_line: int
    confidence: float
    context: str

class LocAgent(Agent):
    """Agent specialized in code localization using graph-based representation."""
    
    def __init__(self, llm, graph_encoder=None):
        super().__init__(llm)
        self.graph_encoder = graph_encoder
        self.code_graph = None
        
    def step(self, state: State) -> Action:
        """Execute one step of code localization.
        
        Args:
            state: Current state containing task information and history
            
        Returns:
            Action: Next action to take
        """
        # Get the current task from state
        current_task = state.get_current_task()
        if not current_task:
            return MessageAction("No active task found")
            
        # Extract code localization query from task
        query = self._extract_localization_query(current_task)
        if not query:
            return MessageAction("Could not extract code localization query from task")
            
        # If we don't have a code graph yet, build it
        if not self.code_graph:
            self.code_graph = self._build_code_graph(state)
            
        # Use graph-based reasoning to find relevant code locations
        locations = self._find_code_locations(query)
        
        # Format and return results
        return self._format_results(locations)
        
    def _extract_localization_query(self, task: str) -> Optional[str]:
        """Extract the code localization query from the task description."""
        # TODO: Implement query extraction logic
        # This should parse the task to identify what code we're looking for
        return task
        
    def _build_code_graph(self, state: State) -> Any:
        """Build a graph representation of the codebase."""
        if self.graph_encoder:
            return self.graph_encoder.encode_codebase(state.workspace_path)
        else:
            # Fallback to basic file scanning if no graph encoder
            return self._build_basic_graph(state.workspace_path)
            
    def _build_basic_graph(self, workspace_path: str) -> Dict[str, Any]:
        """Build a basic graph representation by scanning files."""
        graph = {
            "files": {},
            "dependencies": {},
            "functions": {},
            "classes": {}
        }
        
        # Scan workspace for Python files
        for file_path in Path(workspace_path).rglob("*.py"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = str(file_path.relative_to(workspace_path))
                    
                    # Parse the file content
                    tree = ast.parse(content)
                    
                    # Extract code structure
                    imports = self._extract_imports(tree)
                    functions = self._extract_functions(tree)
                    classes = self._extract_classes(tree)
                    
                    # Store file information
                    graph["files"][relative_path] = {
                        "content": content,
                        "imports": imports,
                        "functions": functions,
                        "classes": classes
                    }
                    
                    # Build dependency graph
                    for imp in imports:
                        if imp not in graph["dependencies"]:
                            graph["dependencies"][imp] = set()
                        graph["dependencies"][imp].add(relative_path)
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        return graph
        
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from code content."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
                    
        return imports
        
    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function definitions from code content."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [decorator.id for decorator in node.decorator_list 
                                 if isinstance(decorator, ast.Name)],
                    "returns": self._get_return_type(node),
                    "docstring": ast.get_docstring(node)
                }
                functions.append(func_info)
                
        return functions
        
    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class definitions from code content."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "bases": [base.id for base in node.bases 
                            if isinstance(base, ast.Name)],
                    "decorators": [decorator.id for decorator in node.decorator_list 
                                 if isinstance(decorator, ast.Name)],
                    "docstring": ast.get_docstring(node),
                    "methods": self._extract_class_methods(node)
                }
                classes.append(class_info)
                
        return classes
        
    def _extract_class_methods(self, class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract methods from a class definition."""
        methods = []
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_info = {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [decorator.id for decorator in node.decorator_list 
                                 if isinstance(decorator, ast.Name)],
                    "returns": self._get_return_type(node),
                    "docstring": ast.get_docstring(node)
                }
                methods.append(method_info)
                
        return methods
        
    def _get_return_type(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation from a function definition."""
        if func_node.returns:
            if isinstance(func_node.returns, ast.Name):
                return func_node.returns.id
            elif isinstance(func_node.returns, ast.Subscript):
                # Handle generic types like List[str]
                if isinstance(func_node.returns.value, ast.Name):
                    return func_node.returns.value.id
        return None
        
    def _find_code_locations(self, query: str) -> List[CodeLocation]:
        """Find relevant code locations using graph-based reasoning."""
        locations = []
        
        # Split query into keywords for matching
        keywords = set(query.lower().split())
        
        # Search through all files in the graph
        for file_path, file_info in self.code_graph["files"].items():
            # Check file content for keyword matches
            content = file_info["content"].lower()
            
            # Search for functions
            for func in file_info["functions"]:
                if self._matches_keywords(func, keywords):
                    locations.append(CodeLocation(
                        file_path=file_path,
                        start_line=func["start_line"],
                        end_line=func["end_line"],
                        confidence=self._calculate_confidence(func, keywords),
                        context=self._get_context(file_info["content"], func["start_line"], func["end_line"])
                    ))
                    
            # Search for classes
            for cls in file_info["classes"]:
                if self._matches_keywords(cls, keywords):
                    locations.append(CodeLocation(
                        file_path=file_path,
                        start_line=cls["start_line"],
                        end_line=cls["end_line"],
                        confidence=self._calculate_confidence(cls, keywords),
                        context=self._get_context(file_info["content"], cls["start_line"], cls["end_line"])
                    ))
                    
                    # Also check class methods
                    for method in cls["methods"]:
                        if self._matches_keywords(method, keywords):
                            locations.append(CodeLocation(
                                file_path=file_path,
                                start_line=method["start_line"],
                                end_line=method["end_line"],
                                confidence=self._calculate_confidence(method, keywords),
                                context=self._get_context(file_info["content"], method["start_line"], method["end_line"])
                            ))
                            
        # Sort locations by confidence
        locations.sort(key=lambda x: x.confidence, reverse=True)
        
        return locations
        
    def _matches_keywords(self, node: Dict[str, Any], keywords: set) -> bool:
        """Check if a node matches any of the keywords."""
        # Check name
        if any(keyword in node["name"].lower() for keyword in keywords):
            return True
            
        # Check docstring
        if node.get("docstring"):
            if any(keyword in node["docstring"].lower() for keyword in keywords):
                return True
                
        # Check arguments
        if "args" in node:
            if any(keyword in " ".join(node["args"]).lower() for keyword in keywords):
                return True
                
        # Check return type
        if node.get("returns"):
            if any(keyword in node["returns"].lower() for keyword in keywords):
                return True
                
        return False
        
    def _calculate_confidence(self, node: Dict[str, Any], keywords: set) -> float:
        """Calculate confidence score for a match."""
        confidence = 0.0
        
        # Name match has highest weight
        if any(keyword in node["name"].lower() for keyword in keywords):
            confidence += 0.5
            
        # Docstring match has second highest weight
        if node.get("docstring"):
            if any(keyword in node["docstring"].lower() for keyword in keywords):
                confidence += 0.3
                
        # Argument match has medium weight
        if "args" in node:
            if any(keyword in " ".join(node["args"]).lower() for keyword in keywords):
                confidence += 0.2
                
        # Return type match has lower weight
        if node.get("returns"):
            if any(keyword in node["returns"].lower() for keyword in keywords):
                confidence += 0.1
                
        return min(confidence, 1.0)
        
    def _get_context(self, content: str, start_line: int, end_line: int, context_lines: int = 3) -> str:
        """Get context around a code location."""
        lines = content.splitlines()
        start = max(0, start_line - context_lines - 1)
        end = min(len(lines), end_line + context_lines)
        
        context = []
        for i in range(start, end):
            prefix = "..." if i < start_line - 1 or i > end_line - 1 else ">"
            context.append(f"{prefix} {lines[i]}")
            
        return "\n".join(context)
        
    def _format_results(self, locations: List[CodeLocation]) -> Action:
        """Format the found code locations into a response."""
        if not locations:
            return MessageAction("No relevant code locations found")
            
        # Format locations into a readable response
        response = "Found relevant code locations:\n\n"
        for loc in locations:
            response += f"File: {loc.file_path}\n"
            response += f"Lines: {loc.start_line}-{loc.end_line}\n"
            response += f"Confidence: {loc.confidence:.2f}\n"
            response += f"Context: {loc.context}\n\n"
            
        return MessageAction(response) 