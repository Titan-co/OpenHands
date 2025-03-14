from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from openhands.controller.agent import Agent
from openhands.events.action import Action, MessageAction
from ..graph_encoder.hero_graph import HeroGraph, HeroNode, NodeType, EdgeType

@dataclass
class ReasoningContext:
    """Context for LLM reasoning about code locations."""
    query: str
    initial_locations: List[HeroNode]
    related_locations: List[HeroNode]
    graph_context: Optional[HeroGraph] = None
    metadata: Optional[Dict[str, Any]] = None

class LLMReasoner:
    """LLM-based reasoning for code localization."""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.prompt_templates = {
            'initial': """Given the following code locations and their context, analyze which ones are most relevant to the query.
Query: {query}

Initial Locations:
{initial_locations}

Related Locations:
{related_locations}

Please analyze the relevance of each location and provide a confidence score (0-1) for each.
Format your response as a JSON list of objects with the following structure:
[
    {{
        "node_id": "string",
        "confidence": float,
        "reasoning": "string"
    }}
]""",
            'context': """Given the following code context and query, analyze the relevance of the code.
Query: {query}

Code Context:
{code_context}

Please analyze the relevance and provide a confidence score (0-1) and reasoning.
Format your response as a JSON object with the following structure:
{{
    "confidence": float,
    "reasoning": "string"
}}"""
        }
        
    def reason(self, context: ReasoningContext) -> List[Dict[str, Any]]:
        """Perform LLM-based reasoning about code locations."""
        # Format initial locations
        initial_locations_text = self._format_locations(context.initial_locations)
        related_locations_text = self._format_locations(context.related_locations)
        
        # Generate initial prompt
        prompt = self.prompt_templates['initial'].format(
            query=context.query,
            initial_locations=initial_locations_text,
            related_locations=related_locations_text
        )
        
        # Get LLM response
        response = self.agent.llm.completion(
            model="gpt-4",  # or other model as configured
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        try:
            # Parse response
            results = json.loads(response.choices[0].message.content)
            
            # If we have graph context, perform additional reasoning
            if context.graph_context:
                results = self._enhance_with_graph_context(results, context)
                
            return results
            
        except json.JSONDecodeError:
            # Fallback to basic confidence scoring
            return self._fallback_reasoning(context)
            
    def _format_locations(self, locations: List[HeroNode]) -> str:
        """Format locations for prompt."""
        formatted = []
        for loc in locations:
            formatted.append(f"Location: {loc.name} ({loc.type.value})")
            if loc.context:
                formatted.append(f"Context:\n{loc.context}")
            formatted.append("---")
        return "\n".join(formatted)
        
    def _enhance_with_graph_context(self, initial_results: List[Dict[str, Any]], 
                                  context: ReasoningContext) -> List[Dict[str, Any]]:
        """Enhance results with graph-based reasoning."""
        enhanced_results = []
        
        for result in initial_results:
            node_id = result["node_id"]
            node = context.graph_context.get_node(node_id)
            
            if not node:
                continue
                
            # Get context subgraph
            subgraph = context.graph_context.get_context_subgraph(node_id)
            
            # Generate context prompt
            context_prompt = self.prompt_templates['context'].format(
                query=context.query,
                code_context=self._format_graph_context(subgraph)
            )
            
            # Get additional reasoning
            response = self.agent.llm.completion(
                model="gpt-4",
                messages=[{"role": "user", "content": context_prompt}],
                temperature=0.2
            )
            
            try:
                context_result = json.loads(response.choices[0].message.content)
                
                # Combine results
                enhanced_results.append({
                    "node_id": node_id,
                    "confidence": (result["confidence"] + context_result["confidence"]) / 2,
                    "reasoning": f"{result['reasoning']}\nAdditional Context: {context_result['reasoning']}"
                })
                
            except json.JSONDecodeError:
                # Keep original result if context reasoning fails
                enhanced_results.append(result)
                
        return enhanced_results
        
    def _format_graph_context(self, subgraph: HeroGraph) -> str:
        """Format graph context for prompt."""
        context_parts = []
        
        # Add nodes
        for node_type in NodeType:
            nodes = subgraph.get_nodes_by_type(node_type)
            if nodes:
                context_parts.append(f"\n{node_type.value.title()}s:")
                for node in nodes:
                    context_parts.append(f"- {node.name}")
                    if node.context:
                        context_parts.append(f"  Context: {node.context}")
                        
        # Add edges
        for edge_type in EdgeType:
            edges = subgraph.get_edges_by_type(edge_type)
            if edges:
                context_parts.append(f"\n{edge_type.value.title()} Relationships:")
                for source, target in edges:
                    source_node = subgraph.get_node(source)
                    target_node = subgraph.get_node(target)
                    if source_node and target_node:
                        context_parts.append(f"- {source_node.name} -> {target_node.name}")
                        
        return "\n".join(context_parts)
        
    def _fallback_reasoning(self, context: ReasoningContext) -> List[Dict[str, Any]]:
        """Fallback reasoning when LLM response parsing fails."""
        results = []
        
        # Simple keyword matching
        query_keywords = set(context.query.lower().split())
        
        for loc in context.initial_locations + context.related_locations:
            # Calculate basic confidence based on keyword matching
            confidence = 0.0
            
            # Check name
            if any(keyword in loc.name.lower() for keyword in query_keywords):
                confidence += 0.5
                
            # Check context
            if loc.context and any(keyword in loc.context.lower() for keyword in query_keywords):
                confidence += 0.3
                
            # Check metadata
            if loc.metadata:
                metadata_text = json.dumps(loc.metadata).lower()
                if any(keyword in metadata_text for keyword in query_keywords):
                    confidence += 0.2
                    
            if confidence > 0:
                results.append({
                    "node_id": loc.id,
                    "confidence": min(confidence, 1.0),
                    "reasoning": "Fallback: Keyword-based matching"
                })
                
        return results 