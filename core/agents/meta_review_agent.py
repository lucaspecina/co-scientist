"""
Meta-Review Agent for AI Co-Scientist

This module implements the Meta-Review Agent, which analyzes patterns in reviews and debates,
provides feedback to other agents, and synthesizes top hypotheses into comprehensive
research overviews with experimental designs and resource planning.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import re
from enum import Enum

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)

class OutputFormat(Enum):
    COMPREHENSIVE = "comprehensive"
    NIH_AIMS = "nih_aims"
    NATURE = "nature"
    SCIENCE = "science"

class MetaReviewAgent(BaseAgent):
    """
    Meta-Review Agent synthesizes insights from reviews and tournament debates.
    
    This agent plays a crucial role in the co-scientist's feedback loop by:
    1. Identifying common patterns in reviews and scientific debates
    2. Providing feedback to improve future reviews and hypothesis generation
    3. Synthesizing top-ranked hypotheses into comprehensive research overviews
    4. Designing detailed experimental approaches with resource estimates
    5. Generating timeline and resource allocation plans
    6. Identifying potential research contacts and collaborators
    7. Supporting multiple output formats for research overviews
    8. Providing quantitative metrics on hypothesis evolution
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the meta-review agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.output_formats = [f.value for f in OutputFormat]
        self.include_contacts = config.get("include_contacts", True)
        self.max_critique_items = config.get("max_critique_items", 10)
        self.max_research_areas = config.get("max_research_areas", 5)
        self.include_resource_estimates = config.get("include_resource_estimates", True)
        self.include_alternative_approaches = config.get("include_alternative_approaches", True)
        self.experiment_detail_level = config.get("experiment_detail_level", "high")
        self.domain_specific_templates = config.get("domain_specific_templates", {})
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate meta-review critique and research overview.
        
        Args:
            context: Dictionary containing:
                - goal: Research goal information
                - hypotheses: List of top hypotheses
                - reviews: List of reviews
                - tournament_results: List of tournament match results
                - iteration: Current iteration number
                - feedback: List of feedback entries (optional)
                - domain: Research domain (optional)
            params: Dictionary containing:
                - task: Task to perform ("critique", "overview", "contacts", "timeline", "all")
                - output_format: Format for research overview
                - timeline_months: Number of months for timeline planning
                
        Returns:
            Dictionary containing:
                - meta_critique: Meta-review critique of patterns in reviews and debates
                - research_overview: Synthesized research overview
                - experimental_design: Detailed experimental design with resource estimates
                - timeline: Research timeline and resource allocation plan
                - research_contacts: Suggested research contacts (if requested)
        """
        goal = context.get("goal", {})
        hypotheses = context.get("hypotheses", [])
        reviews = context.get("reviews", [])
        tournament_results = context.get("tournament_results", [])
        iteration = context.get("iteration", 0)
        feedback = context.get("feedback", [])
        domain = context.get("domain", "")
        
        # Extract parameters
        task = params.get("task", "all")
        output_format = params.get("output_format", self.output_formats[0])
        timeline_months = params.get("timeline_months", 12)
        
        result = {}
        
        try:
            # Generate meta-review critique with quantitative metrics
            if task in ["critique", "all"]:
                meta_critique = await self._generate_meta_critique(
                    goal=goal,
                    reviews=reviews,
                    tournament_results=tournament_results,
                    iteration=iteration,
                    domain=domain
                )
                result["meta_critique"] = meta_critique
            
            # Generate research overview with experimental design
            if task in ["overview", "all"]:
                research_overview = await self._generate_research_overview(
                    goal=goal,
                    hypotheses=hypotheses,
                    reviews=reviews,
                    iteration=iteration,
                    feedback=feedback,
                    output_format=output_format,
                    domain=domain
                )
                result["research_overview"] = research_overview
                
                # Generate detailed experimental design with resource estimates
                if self.include_resource_estimates:
                    experimental_design = await self._generate_experimental_design(
                        goal=goal,
                        hypotheses=hypotheses,
                        domain=domain
                    )
                    result["experimental_design"] = experimental_design
            
            # Generate timeline and resource allocation plan
            if task in ["timeline", "all"]:
                timeline = await self._generate_timeline(
                    goal=goal,
                    hypotheses=hypotheses,
                    experimental_design=result.get("experimental_design", {}),
                    timeline_months=timeline_months
                )
                result["timeline"] = timeline
            
            # Identify research contacts
            if task in ["contacts", "all"] and self.include_contacts:
                research_contacts = await self._identify_research_contacts(
                    goal=goal,
                    hypotheses=hypotheses,
                    reviews=reviews,
                    domain=domain
                )
                result["research_contacts"] = research_contacts
            
            return result
        except Exception as e:
            logger.error(f"Error in meta-review agent: {str(e)}")
            raise AgentExecutionError(f"Failed to generate meta-review: {str(e)}")
    
    async def _generate_meta_critique(self,
                               goal: Dict[str, Any],
                               reviews: List[Dict[str, Any]],
                               tournament_results: List[Dict[str, Any]],
                               iteration: int,
                               domain: str) -> Dict[str, Any]:
        """
        Generate a meta-review critique identifying patterns in reviews and debates.
        
        Args:
            goal: Research goal information
            reviews: List of reviews
            tournament_results: List of tournament match results
            iteration: Current iteration number
            domain: Research domain
            
        Returns:
            Dictionary containing meta-critique information with quantitative metrics
        """
        logger.info("Generating meta-review critique")
        
        if not reviews and not tournament_results:
            logger.warning("No reviews or tournament results available for meta-critique")
            return {
                "common_issues": [],
                "improvement_suggestions": [],
                "metrics": {},
                "iteration": iteration
            }
        
        system_prompt = """
        You are a scientific meta-reviewer analyzing patterns across multiple scientific reviews and debates.
        Your task is to identify recurring issues, patterns, and opportunities for improvement in the scientific reasoning process.
        Focus on being specific, actionable, and insightful in your analysis.
        Include quantitative metrics where possible.
        """
        
        # Extract review critiques and tournament debate rationales
        review_critiques = []
        for review in reviews:
            critique = review.get("critique", "")
            if critique:
                review_critiques.append(critique)
        
        tournament_rationales = []
        for result in tournament_results:
            rationale = result.get("rationale", "")
            if rationale:
                tournament_rationales.append(rationale)
        
        # Use domain-specific template if available
        template = self.domain_specific_templates.get(domain, {}).get("meta_critique", "")
        if template:
            prompt = template.format(
                goal=goal.get("description", ""),
                domain=domain,
                review_critiques=json.dumps(review_critiques[:20]),
                tournament_rationales=json.dumps(tournament_rationales[:20]),
                max_items=self.max_critique_items
            )
        else:
            prompt = f"""
            RESEARCH GOAL: {goal.get('description', '')}
            
            DOMAIN: {domain}
            
            TASK: Analyze the following scientific reviews and tournament debate rationales to identify:
            1. Common issues, errors, or limitations found across multiple reviews/debates
            2. Recurring patterns in how hypotheses are evaluated
            3. Specific suggestions for improving future hypothesis generation and evaluation
            4. Quantitative metrics on review quality and consistency
            
            REVIEW CRITIQUES ({len(review_critiques)}):
            {json.dumps(review_critiques[:20]) if review_critiques else "No review critiques available."}
            
            TOURNAMENT DEBATE RATIONALES ({len(tournament_rationales)}):
            {json.dumps(tournament_rationales[:20]) if tournament_rationales else "No tournament rationales available."}
            
            Please identify the {self.max_critique_items} most important patterns and issues, and provide specific suggestions for improvement.
            Format your response as a JSON object with the following structure:
            {{
                "common_issues": [
                    {{
                        "issue": "Brief description of the issue",
                        "examples": ["Example 1", "Example 2"],
                        "impact": "How this affects hypothesis quality",
                        "frequency": "Percentage of reviews showing this issue"
                    }}
                ],
                "improvement_suggestions": [
                    {{
                        "target": "generation|reflection|ranking|evolution",
                        "suggestion": "Specific suggestion for improvement",
                        "rationale": "Why this would help",
                        "expected_impact": "Quantitative estimate of improvement"
                    }}
                ],
                "metrics": {{
                    "review_consistency": "0-1 score measuring review consistency",
                    "critique_depth": "Average depth of critiques (1-5)",
                    "rationale_quality": "Average quality of debate rationales (1-5)",
                    "hypothesis_evolution": "Rate of hypothesis improvement over iterations"
                }}
            }}
            """
        
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                critique_data = json.loads(json_str)
            else:
                logger.warning("No JSON object found in meta-critique response")
                critique_data = {
                    "common_issues": [],
                    "improvement_suggestions": [],
                    "metrics": {}
                }
            
            # Add iteration information
            critique_data["iteration"] = iteration
            
            return critique_data
        except Exception as e:
            logger.error(f"Error parsing meta-critique response: {str(e)}")
            return {
                "common_issues": [],
                "improvement_suggestions": [],
                "metrics": {},
                "iteration": iteration,
                "error": str(e)
            }
    
    async def _generate_experimental_design(self,
                                    goal: Dict[str, Any],
                                    hypotheses: List[Dict[str, Any]],
                                    domain: str) -> Dict[str, Any]:
        """
        Generate detailed experimental design with resource estimates.
        
        Args:
            goal: Research goal information
            hypotheses: List of hypotheses
            domain: Research domain
            
        Returns:
            Dictionary containing experimental design information
        """
        logger.info("Generating experimental design with resource estimates")
        
        if not hypotheses:
            logger.warning("No hypotheses available for experimental design")
            return {
                "experiments": [],
                "resource_estimates": {},
                "alternative_approaches": []
            }
        
        system_prompt = """
        You are a scientific experiment designer with expertise in resource planning.
        Your task is to design detailed experiments to test the hypotheses, including resource estimates and alternative approaches.
        Focus on practical feasibility while maintaining scientific rigor.
        """
        
        # Sort hypotheses by score
        sorted_hypotheses = sorted(
            hypotheses,
            key=lambda h: h.get("score", 0),
            reverse=True
        )[:5]  # Focus on top 5 hypotheses
        
        # Use domain-specific template if available
        template = self.domain_specific_templates.get(domain, {}).get("experimental_design", "")
        if template:
            prompt = template.format(
                goal=goal.get("description", ""),
                domain=domain,
                hypotheses=json.dumps([{
                    "text": h.get("text", ""),
                    "rationale": h.get("rationale", ""),
                    "score": h.get("score", 0)
                } for h in sorted_hypotheses]),
                detail_level=self.experiment_detail_level
            )
        else:
            prompt = f"""
            RESEARCH GOAL: {goal.get('description', '')}
            
            DOMAIN: {domain}
            
            TOP HYPOTHESES:
            {json.dumps([{
                "text": h.get("text", ""),
                "rationale": h.get("rationale", ""),
                "score": h.get("score", 0)
            } for h in sorted_hypotheses])}
            
            TASK: Design detailed experiments to test these hypotheses, including:
            1. For each hypothesis:
               - Primary experimental approach
               - Control experiments
               - Validation methods
               - Expected outcomes and interpretation
               - Potential pitfalls and mitigation strategies
            2. Resource estimates for each experiment:
               - Required equipment and materials
               - Estimated time requirements
               - Personnel needs and expertise required
               - Approximate cost ranges
            3. Alternative experimental approaches:
               - High-risk, high-reward alternatives
               - Lower-cost pilot studies
               - Complementary approaches
            
            Detail Level: {self.experiment_detail_level}
            
            Format your response as a JSON object with the following structure:
            {{
                "experiments": [
                    {{
                        "hypothesis_id": "ID of the hypothesis",
                        "primary_approach": {{
                            "description": "Detailed experimental procedure",
                            "controls": ["Control experiment 1", "Control experiment 2"],
                            "validation": ["Validation method 1", "Validation method 2"],
                            "expected_outcomes": {{
                                "positive": "Expected results if hypothesis is correct",
                                "negative": "Alternative outcomes to consider"
                            }},
                            "pitfalls": [
                                {{
                                    "issue": "Potential problem",
                                    "mitigation": "How to address it"
                                }}
                            ]
                        }},
                        "resource_requirements": {{
                            "equipment": ["Item 1", "Item 2"],
                            "materials": ["Material 1", "Material 2"],
                            "time_estimate": "Estimated duration",
                            "personnel": [
                                {{
                                    "role": "Required role",
                                    "expertise": "Required expertise",
                                    "time_commitment": "FTE or hours"
                                }}
                            ],
                            "estimated_costs": {{
                                "min": "Minimum cost estimate",
                                "max": "Maximum cost estimate",
                                "currency": "USD"
                            }}
                        }}
                    }}
                ],
                "alternative_approaches": [
                    {{
                        "type": "high_risk|pilot|complementary",
                        "description": "Description of the approach",
                        "advantages": ["Advantage 1", "Advantage 2"],
                        "disadvantages": ["Disadvantage 1", "Disadvantage 2"],
                        "resource_impact": "How this affects resource requirements"
                    }}
                ]
            }}
            """
        
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                design_data = json.loads(json_str)
            else:
                logger.warning("No JSON object found in experimental design response")
                design_data = {
                    "experiments": [],
                    "alternative_approaches": []
                }
            
            return design_data
        except Exception as e:
            logger.error(f"Error parsing experimental design response: {str(e)}")
            return {
                "experiments": [],
                "alternative_approaches": [],
                "error": str(e)
            }
    
    async def _generate_timeline(self,
                          goal: Dict[str, Any],
                          hypotheses: List[Dict[str, Any]],
                          experimental_design: Dict[str, Any],
                          timeline_months: int) -> Dict[str, Any]:
        """
        Generate research timeline and resource allocation plan.
        
        Args:
            goal: Research goal information
            hypotheses: List of hypotheses
            experimental_design: Experimental design information
            timeline_months: Number of months to plan for
            
        Returns:
            Dictionary containing timeline and resource allocation information
        """
        logger.info(f"Generating {timeline_months}-month research timeline")
        
        if not experimental_design.get("experiments"):
            logger.warning("No experiments available for timeline planning")
            return {
                "timeline": [],
                "resource_allocation": {},
                "milestones": []
            }
        
        system_prompt = """
        You are a research project manager with expertise in timeline planning and resource allocation.
        Your task is to create a detailed research timeline with clear milestones and resource allocation plans.
        Focus on efficient use of resources while maintaining scientific quality.
        """
        
        prompt = f"""
        RESEARCH GOAL: {goal.get('description', '')}
        
        EXPERIMENTAL DESIGN:
        {json.dumps(experimental_design)}
        
        TIMELINE DURATION: {timeline_months} months
        
        TASK: Create a detailed research timeline and resource allocation plan that:
        1. Organizes experiments into logical phases
        2. Identifies critical path and dependencies
        3. Sets clear milestones and deliverables
        4. Allocates resources efficiently
        5. Includes contingency planning
        
        Format your response as a JSON object with the following structure:
        {{
            "timeline": [
                {{
                    "phase": "Phase name",
                    "start_month": 1,
                    "duration_months": 3,
                    "experiments": ["Experiment 1", "Experiment 2"],
                    "deliverables": ["Deliverable 1", "Deliverable 2"],
                    "dependencies": ["Previous phase name"],
                    "resource_requirements": {{
                        "personnel": ["Role 1", "Role 2"],
                        "equipment": ["Equipment 1", "Equipment 2"],
                        "materials": ["Material 1", "Material 2"]
                    }}
                }}
            ],
            "resource_allocation": {{
                "personnel": [
                    {{
                        "role": "Role name",
                        "allocation": [
                            {{
                                "phase": "Phase name",
                                "commitment": "FTE or hours"
                            }}
                        ]
                    }}
                ],
                "equipment": [
                    {{
                        "name": "Equipment name",
                        "allocation": [
                            {{
                                "phase": "Phase name",
                                "usage": "Usage details"
                            }}
                        ]
                    }}
                ]
            }},
            "milestones": [
                {{
                    "name": "Milestone name",
                    "month": 3,
                    "deliverables": ["Deliverable 1", "Deliverable 2"],
                    "success_criteria": ["Criterion 1", "Criterion 2"]
                }}
            ],
            "contingency_plans": [
                {{
                    "trigger": "What triggers this plan",
                    "actions": ["Action 1", "Action 2"],
                    "resource_impact": "Impact on timeline and resources"
                }}
            ]
        }}
        """
        
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                timeline_data = json.loads(json_str)
            else:
                logger.warning("No JSON object found in timeline response")
                timeline_data = {
                    "timeline": [],
                    "resource_allocation": {},
                    "milestones": [],
                    "contingency_plans": []
                }
            
            return timeline_data
        except Exception as e:
            logger.error(f"Error parsing timeline response: {str(e)}")
            return {
                "timeline": [],
                "resource_allocation": {},
                "milestones": [],
                "contingency_plans": [],
                "error": str(e)
            }
    
    async def _generate_research_overview(self,
                                   goal: Dict[str, Any],
                                   hypotheses: List[Dict[str, Any]],
                                   reviews: List[Dict[str, Any]],
                                   iteration: int,
                                   feedback: List[Dict[str, Any]],
                                   output_format: str,
                                   domain: str) -> Dict[str, Any]:
        """
        Generate a comprehensive research overview based on top hypotheses.
        
        Args:
            goal: Research goal information
            hypotheses: List of top hypotheses
            reviews: List of reviews
            iteration: Current iteration number
            feedback: List of feedback entries
            output_format: Format for the research overview
            domain: Research domain
            
        Returns:
            Dictionary containing research overview information
        """
        logger.info(f"Generating research overview in {output_format} format")
        
        if not hypotheses:
            logger.warning("No hypotheses available for research overview")
            return {
                "title": "Research Overview (No Hypotheses Available)",
                "summary": "No hypotheses are available to generate a research overview.",
                "research_areas": [],
                "format": output_format,
                "iteration": iteration
            }
        
        # Sort hypotheses by score (descending)
        sorted_hypotheses = sorted(
            hypotheses, 
            key=lambda h: h.get("score", 0),
            reverse=True
        )
        
        # Take top hypotheses (up to 10)
        top_hypotheses = sorted_hypotheses[:10]
        
        system_prompt = """
        You are a scientific research director synthesizing research hypotheses into a comprehensive research overview.
        Your task is to identify key research areas, organize related hypotheses, and provide a roadmap for future research.
        Be specific, insightful, and focus on scientific value and novelty.
        """
        
        # Build the prompt based on the requested format
        if output_format == "nih_aims":
            prompt = self._build_nih_aims_prompt(goal, top_hypotheses, reviews, feedback, iteration)
        else:  # comprehensive format
            prompt = self._build_comprehensive_overview_prompt(goal, top_hypotheses, reviews, feedback, iteration)
        
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        
        # Process the response based on format
        if output_format == "nih_aims":
            overview = self._parse_nih_aims_response(response, iteration)
        else:
            overview = self._parse_comprehensive_overview_response(response, iteration)
        
        # Add format information
        overview["format"] = output_format
        
        return overview
    
    def _build_comprehensive_overview_prompt(self,
                                       goal: Dict[str, Any],
                                       hypotheses: List[Dict[str, Any]],
                                       reviews: List[Dict[str, Any]],
                                       feedback: List[Dict[str, Any]],
                                       iteration: int) -> str:
        """
        Build prompt for comprehensive research overview.
        
        Args:
            goal: Research goal information
            hypotheses: List of top hypotheses
            reviews: List of reviews
            feedback: List of feedback entries
            iteration: Current iteration number
            
        Returns:
            Formatted prompt string
        """
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        background = goal.get("background", "")
        
        prompt = f"""
        RESEARCH GOAL: {goal_description}
        
        DOMAIN: {domain}
        
        BACKGROUND: {background}
        
        TOP HYPOTHESES:
        """
        
        for i, h in enumerate(hypotheses, 1):
            prompt += f"""
            HYPOTHESIS {i}: {h.get('text', '')}
            RATIONALE: {h.get('rationale', '')}
            SCORE: {h.get('score', 0)}
            """
        
        # Add recent feedback if available
        if feedback:
            prompt += "\nSCIENTIST FEEDBACK:\n"
            recent_feedback = sorted(
                feedback, 
                key=lambda x: x.get("iteration", 0),
                reverse=True
            )[:2]  # Only the 2 most recent feedback entries
            
            for entry in recent_feedback:
                feedback_text = entry.get("text", "")
                feedback_iter = entry.get("iteration", 0)
                
                if feedback_text:
                    prompt += f"From iteration {feedback_iter}: {feedback_text}\n\n"
        
        prompt += f"""
        TASK: Based on the research goal and top hypotheses, create a comprehensive research overview that:
        
        1. Synthesizes the most promising research directions
        2. Identifies {self.max_research_areas} key research areas to explore
        3. For each research area:
           - Provides a clear description of the area
           - Explains its importance to the research goal
           - Suggests 2-3 specific experiments or approaches
           - Lists 2-3 example topics within this area
        4. Concludes with an overall assessment of the current state of knowledge and next steps
        
        Format your response as a structured research overview with clear sections and subsections.
        Focus on scientific value, novelty, and practical feasibility.
        """
        
        return prompt
    
    def _build_nih_aims_prompt(self,
                         goal: Dict[str, Any],
                         hypotheses: List[Dict[str, Any]],
                         reviews: List[Dict[str, Any]],
                         feedback: List[Dict[str, Any]],
                         iteration: int) -> str:
        """
        Build prompt for NIH Specific Aims Page format.
        
        Args:
            goal: Research goal information
            hypotheses: List of top hypotheses
            reviews: List of reviews
            feedback: List of feedback entries
            iteration: Current iteration number
            
        Returns:
            Formatted prompt string
        """
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        background = goal.get("background", "")
        
        prompt = f"""
        RESEARCH GOAL: {goal_description}
        
        DOMAIN: {domain}
        
        BACKGROUND: {background}
        
        TOP HYPOTHESES:
        """
        
        for i, h in enumerate(hypotheses, 1):
            prompt += f"""
            HYPOTHESIS {i}: {h.get('text', '')}
            RATIONALE: {h.get('rationale', '')}
            SCORE: {h.get('score', 0)}
            """
        
        # Add recent feedback if available
        if feedback:
            prompt += "\nSCIENTIST FEEDBACK:\n"
            recent_feedback = sorted(
                feedback, 
                key=lambda x: x.get("iteration", 0),
                reverse=True
            )[:2]  # Only the 2 most recent feedback entries
            
            for entry in recent_feedback:
                feedback_text = entry.get("text", "")
                feedback_iter = entry.get("iteration", 0)
                
                if feedback_text:
                    prompt += f"From iteration {feedback_iter}: {feedback_text}\n\n"
        
        prompt += f"""
        TASK: Based on the research goal and top hypotheses, create a NIH-style Specific Aims Page that:
        
        1. Begins with a compelling introduction paragraph that establishes the significance of the research
        2. Presents the central hypothesis and overall objective
        3. Lists 2-3 specific aims, each with:
           - A clear, testable hypothesis
           - Brief description of the approach
           - Expected outcomes
        4. Concludes with an impact statement explaining the significance and innovation
        
        Format your response following the NIH Specific Aims Page structure (1-page summary).
        Use clear, concise language appropriate for a grant proposal.
        Focus on scientific significance, innovation, and feasibility.
        """
        
        return prompt
    
    def _parse_comprehensive_overview_response(self, response: str, iteration: int) -> Dict[str, Any]:
        """
        Parse response for comprehensive research overview.
        
        Args:
            response: Model response
            iteration: Current iteration number
            
        Returns:
            Parsed overview dictionary
        """
        # Extract title (first heading)
        title_match = re.search(r'^#\s+(.+)$', response, re.MULTILINE)
        title = title_match.group(1) if title_match else "Research Overview"
        
        # Extract research areas
        research_areas = []
        area_pattern = r'#+\s+Research Area\s*\d*\s*:?\s*(.+?)\n([\s\S]*?)(?=#+\s+Research Area|$)'
        area_matches = re.finditer(area_pattern, response, re.IGNORECASE)
        
        for match in area_matches:
            area_title = match.group(1).strip()
            area_content = match.group(2).strip()
            
            # Extract experiments
            experiments = []
            exp_pattern = r'(?:Experiment|Approach)\s*\d*\s*:?\s*(.+?)(?:\n|$)'
            exp_matches = re.finditer(exp_pattern, area_content, re.IGNORECASE)
            for exp_match in exp_matches:
                experiments.append(exp_match.group(1).strip())
            
            # Extract topics
            topics = []
            topic_pattern = r'(?:Topic|Example)\s*\d*\s*:?\s*(.+?)(?:\n|$)'
            topic_matches = re.finditer(topic_pattern, area_content, re.IGNORECASE)
            for topic_match in topic_matches:
                topics.append(topic_match.group(1).strip())
            
            research_areas.append({
                "title": area_title,
                "description": area_content,
                "experiments": experiments,
                "topics": topics
            })
        
        # Extract summary (text before first research area)
        summary_match = re.search(r'^([\s\S]*?)(?=#+\s+Research Area)', response, re.MULTILINE)
        summary = summary_match.group(1).strip() if summary_match else ""
        
        # If no summary found, use the first paragraph
        if not summary:
            paragraphs = response.split('\n\n')
            if paragraphs:
                summary = paragraphs[0].strip()
        
        return {
            "title": title,
            "summary": summary,
            "research_areas": research_areas,
            "full_text": response,
            "iteration": iteration
        }
    
    def _parse_nih_aims_response(self, response: str, iteration: int) -> Dict[str, Any]:
        """
        Parse response for NIH Specific Aims Page format.
        
        Args:
            response: Model response
            iteration: Current iteration number
            
        Returns:
            Parsed overview dictionary
        """
        # Extract title (first heading)
        title_match = re.search(r'^#\s+(.+)$', response, re.MULTILINE)
        title = title_match.group(1) if title_match else "Specific Aims"
        
        # Extract central hypothesis
        hypothesis_match = re.search(r'(?:Central|Overall)\s+Hypothesis:?\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        central_hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else ""
        
        # Extract specific aims
        aims = []
        aim_pattern = r'(?:Specific|Aim)\s+Aim\s*\d*\s*:?\s*(.+?)\n([\s\S]*?)(?=(?:Specific|Aim)\s+Aim|Impact|Significance|$)'
        aim_matches = re.finditer(aim_pattern, response, re.IGNORECASE)
        
        for match in aim_matches:
            aim_title = match.group(1).strip()
            aim_content = match.group(2).strip()
            
            # Extract hypothesis for this aim
            aim_hyp_match = re.search(r'Hypothesis:?\s*(.+?)(?:\n|$)', aim_content, re.IGNORECASE)
            aim_hypothesis = aim_hyp_match.group(1).strip() if aim_hyp_match else ""
            
            # Extract approach
            approach_match = re.search(r'Approach:?\s*([\s\S]*?)(?=Hypothesis|Expected|$)', aim_content, re.IGNORECASE)
            approach = approach_match.group(1).strip() if approach_match else ""
            
            # Extract expected outcomes
            outcomes_match = re.search(r'Expected\s+Outcomes?:?\s*([\s\S]*?)(?=Hypothesis|Approach|$)', aim_content, re.IGNORECASE)
            outcomes = outcomes_match.group(1).strip() if outcomes_match else ""
            
            aims.append({
                "title": aim_title,
                "hypothesis": aim_hypothesis,
                "approach": approach,
                "expected_outcomes": outcomes,
                "full_text": aim_content
            })
        
        # Extract impact statement
        impact_match = re.search(r'(?:Impact|Significance):?\s*([\s\S]*?)$', response, re.IGNORECASE)
        impact = impact_match.group(1).strip() if impact_match else ""
        
        # Extract introduction (text before first aim)
        intro_match = re.search(r'^([\s\S]*?)(?=(?:Specific|Aim)\s+Aim)', response, re.MULTILINE)
        introduction = intro_match.group(1).strip() if intro_match else ""
        
        return {
            "title": title,
            "introduction": introduction,
            "central_hypothesis": central_hypothesis,
            "specific_aims": aims,
            "impact_statement": impact,
            "full_text": response,
            "iteration": iteration
        }
    
    async def _identify_research_contacts(self,
                                   goal: Dict[str, Any],
                                   hypotheses: List[Dict[str, Any]],
                                   reviews: List[Dict[str, Any]],
                                   domain: str) -> List[Dict[str, Any]]:
        """
        Identify potential research contacts and collaborators.
        
        Args:
            goal: Research goal information
            hypotheses: List of top hypotheses
            reviews: List of reviews
            domain: Research domain
            
        Returns:
            List of potential research contacts
        """
        logger.info("Identifying potential research contacts")
        
        if not hypotheses:
            logger.warning("No hypotheses available for identifying research contacts")
            return []
        
        system_prompt = """
        You are a scientific network analyst identifying potential research collaborators.
        Your task is to suggest qualified domain experts who could review or collaborate on specific research hypotheses.
        Focus on identifying researchers with relevant expertise, providing clear rationale for each suggestion.
        """
        
        # Take top 3 hypotheses
        top_hypotheses = sorted(
            hypotheses, 
            key=lambda h: h.get("score", 0),
            reverse=True
        )[:3]
        
        prompt = f"""
        RESEARCH GOAL: {goal.get('description', '')}
        
        DOMAIN: {domain}
        
        TOP HYPOTHESES:
        """
        
        for i, h in enumerate(top_hypotheses, 1):
            prompt += f"""
            HYPOTHESIS {i}: {h.get('text', '')}
            RATIONALE: {h.get('rationale', '')}
            """
        
        prompt += """
        TASK: Based on the research goal and hypotheses, identify 3-5 potential research contacts who would be qualified to review or collaborate on this research.
        
        For each contact, provide:
        1. Name and affiliation
        2. Area of expertise
        3. Specific reason why they would be valuable for this research
        4. Potential contribution they could make
        
        Format your response as a JSON array of contact objects with the following structure:
        [
            {
                "name": "Researcher name",
                "affiliation": "University/Institution",
                "expertise": "Brief description of expertise",
                "relevance": "Why they are relevant to this research",
                "potential_contribution": "How they could contribute"
            }
        ]
        
        Focus on identifying researchers with directly relevant expertise to the specific hypotheses.
        """
        
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                json_str = json_match.group(0)
                contacts = json.loads(json_str)
            else:
                logger.warning("No JSON array found in research contacts response")
                contacts = []
            
            return contacts
        except Exception as e:
            logger.error(f"Error parsing research contacts response: {str(e)}")
            return []

    async def _analyze_hypothesis_evolution(self,
                                    hypotheses: List[Dict[str, Any]],
                                    reviews: List[Dict[str, Any]],
                                    tournament_results: List[Dict[str, Any]],
                                    iteration: int) -> Dict[str, Any]:
        """Analyze hypothesis evolution patterns and metrics."""
        logger.info("Analyzing hypothesis evolution")
        
        if not hypotheses:
            return {
                "evolution_rate": 0.0,
                "quality_trends": {},
                "diversity_metrics": {},
                "iteration": iteration
            }
        
        system_prompt = """
        You are a scientific metrics analyst evaluating hypothesis evolution.
        Your task is to analyze patterns in how hypotheses have evolved and improved over iterations.
        Focus on quantitative metrics and clear trends.
        """
        
        prompt = f"""
        HYPOTHESES OVER TIME:
        {json.dumps([{
            "text": h.get("text", ""),
            "score": h.get("score", 0),
            "iteration": h.get("iteration", 0)
        } for h in hypotheses])}
        
        TOURNAMENT RESULTS:
        {json.dumps(tournament_results)}
        
        TASK: Analyze the evolution of hypotheses by calculating:
        1. Rate of hypothesis improvement (score changes over iterations)
        2. Diversity metrics (semantic similarity between hypotheses)
        3. Quality trends (changes in specific evaluation criteria)
        4. Tournament performance patterns
        
        Format your response as a JSON object with the following structure:
        {{
            "evolution_rate": "Average improvement in hypothesis scores per iteration",
            "quality_trends": {{
                "novelty": "Trend in novelty scores",
                "plausibility": "Trend in plausibility scores",
                "testability": "Trend in testability scores"
            }},
            "diversity_metrics": {{
                "semantic_diversity": "Measure of hypothesis diversity",
                "mechanism_diversity": "Diversity in proposed mechanisms"
            }},
            "tournament_patterns": [
                {{
                    "pattern": "Identified pattern",
                    "frequency": "How often it occurs",
                    "significance": "Impact on hypothesis quality"
                }}
            ]
        }}
        """
        
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                evolution_data = json.loads(json_match.group(0))
                evolution_data["iteration"] = iteration
                return evolution_data
            else:
                logger.warning("No JSON object found in evolution analysis response")
                return {
                    "evolution_rate": 0.0,
                    "quality_trends": {},
                    "diversity_metrics": {},
                    "iteration": iteration
                }
        except Exception as e:
            logger.error(f"Error parsing evolution analysis response: {str(e)}")
            return {
                "evolution_rate": 0.0,
                "quality_trends": {},
                "diversity_metrics": {},
                "iteration": iteration,
                "error": str(e)
            } 