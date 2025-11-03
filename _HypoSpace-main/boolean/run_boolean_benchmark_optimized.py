import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
import ast
import sympy as sp
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
from textwrap import dedent
from collections import Counter, defaultdict
from scipy import stats
import traceback

# --- Self-consistency helpers ---
def _boolean_mismatch_score(expr_str: str, observations: List['BooleanObservation'], variables: List[str], operators: Set[str]) -> int:
    try:
        expr = BooleanExpression(expr_str, variables, operators)
        if expr.sympy_expr is None:
            return 10**9
        score = 0
        for obs in observations:
            try:
                pred = expr.evaluate(obs.inputs)
                score += 0 if pred == obs.output else 1
            except Exception:
                score += 1
        return score
    except Exception:
        return 10**9


def _expr_depth_and_ops(expr_str: str, variables: List[str], operators: Set[str]) -> Tuple[int, int]:
    try:
        expr = BooleanExpression(expr_str, variables, operators)
        if expr.sympy_expr is None:
            return (10**6, 10**6)
        # depth
        def _depth(node) -> int:
            if node is None:
                return 0
            if isinstance(node, (sp.Symbol, sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                return 0
            if hasattr(node, 'args') and node.args:
                return 1 + max(_depth(a) for a in node.args)
            return 0
        d = _depth(expr.sympy_expr)
        # op count
        cnt = 0
        for node in sp.preorder_traversal(expr.sympy_expr):
            if isinstance(node, (sp.And, sp.Or, sp.Not, sp.Xor)) or (isinstance(node, sp.Not) and isinstance(node.args[0], sp.Or)):
                cnt += 1
        return (d, cnt)
    except Exception:
        return (10**6, 10**6)

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent))

from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM
from boolean_dataset import BooleanExpression, BooleanObservation


class BooleanBenchmarkRefined:
    def __init__(self, complete_dataset_path: str):
        """
        Initialize benchmark with a complete dataset.

        Args:
            complete_dataset_path: Path to the complete Boolean dataset JSON file
        """
        with open(complete_dataset_path, 'r') as f:
            self.complete_dataset = json.load(f)

        # Extract metadata
        self.mechanistic_opts = self.complete_dataset['metadata']['mechanistic_opts']
        self.metadata = self.complete_dataset['metadata']
        self.variables = self.metadata['variables']
        self.operators = set(self.metadata['operators'])
        self.max_depth = self.metadata['max_depth']

        # Flatten all datasets into a single list for sampling
        self.all_observation_sets = []
        for n_obs, datasets in self.complete_dataset['datasets_by_n_observations'].items():
            self.all_observation_sets.extend(datasets)

        print(f"Loaded complete dataset with {len(self.all_observation_sets)} observation sets")
        print(f"Variables: {', '.join(self.variables)}")
        print(f"Operators: {', '.join(self.operators)}")
        print(f"Max depth: {self.max_depth}")

    def sample_observation_sets(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """
        Sample n observation sets from the complete dataset.

        Args:
            n_samples: Number of observation sets to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled observation sets
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Sample without replacement
        n_available = len(self.all_observation_sets)
        n_to_sample = min(n_samples, n_available)

        if n_to_sample < n_samples:
            print(f"Warning: Requested {n_samples} samples but only {n_available} available")

        sampled = random.sample(self.all_observation_sets, n_to_sample)
        return sampled

    def create_prompt(self, observations: List[BooleanObservation],
                      prior_hypotheses: List[str]) -> str:
        # Format observations
        obs_lines = []
        for obs in observations:
            obs_lines.append(obs.to_string())
        obs_block = "\n".join(obs_lines)

        # Format prior hypotheses
        if prior_hypotheses:
            prior_block = "\n".join([f"Expression: {h}" for h in prior_hypotheses])
        else:
            prior_block = "None"

        # Build operator description
        op_descriptions = []
        if 'AND' in self.operators: op_descriptions.append("AND (conjunction)")
        if 'OR' in self.operators: op_descriptions.append("OR (disjunction)")
        if 'NOT' in self.operators: op_descriptions.append("NOT (negation)")
        if 'XOR' in self.operators: op_descriptions.append("XOR (exclusive or)")
        if 'NOR' in self.operators: op_descriptions.append("NOR (NOT (x OR y))")

        operators_str = ", ".join(op_descriptions)

        # Pull mechanistic opts from the loaded dataset
        mech = self.complete_dataset.get('metadata', {}).get('mechanistic_opts', {})
        comm = mech.get('apply_commutativity', True)
        idem = mech.get('apply_idempotence_and_or', True)
        flat = mech.get('flatten_associativity', True)

        eq_rules = []
        if comm:
            eq_rules.append(
                "- Commutativity: reorderings are the SAME (e.g., x AND y = y AND x)."
            )
        if idem:
            eq_rules.append(
                "- Idempotence: duplicates of the same input under AND/OR collapse (e.g., x AND x = x; x OR x = x)."
            )
        if flat:
            eq_rules.append(
                "- Associativity flattening: cascades of the SAME operator are the SAME regardless of parentheses "
                "(e.g., (x AND y) AND x = x AND (y AND x) = x AND x AND y)."
            )
        else:
            eq_rules.append(
                "- No associativity flattening: different parenthesizations of the SAME operator are DIFFERENT "
                "(e.g., (x AND y) AND x ≠ x AND (y AND x))."
            )

        eq_rules.append(
            "- No distributivity, absorption, or De Morgan normalization: mixing operators keeps expressions DIFFERENT "
            "(e.g., (x AND y) OR z ≠ x AND (y OR z))."
        )

        rules_text = "\n        ".join(eq_rules)

        # Add explicit diversity encouragement
        diversity_prompt = ""
        if prior_hypotheses:
            diversity_prompt = f"""
        7. Prior expressions show these patterns: {self._extract_patterns(prior_hypotheses)}. 
           Generate expressions with DIFFERENT structural patterns (e.g., different operator combinations,
           varying nesting structures, or alternative variable groupings).
            """

        prompt = f"""You are given partial observations of a Boolean function with variables: {', '.join(self.variables)}

            Allowed operators: {operators_str}

            Observations (input -> output):
            {obs_block}

            Prior expressions generated (avoid repeating any expression 
            that is equivalent under the rules below):
            {prior_block}

            Task: Generate a single Boolean expression that is consistent with ALL observations.

            Requirements:
            1. Use ONLY the variables: {', '.join(self.variables)}
            2. Use ONLY these operators: {', '.join(self.operators)}
            3. The expression must match all given observations
            4. Structural uniqueness is judged by these rules:
            {rules_text}
            5. Expression depth should be at most {self.max_depth} levels of nesting
            6. Do not use boolean constants True or False anywhere in the expression.
            {diversity_prompt}

            Output format: 
            - Return ONLY the Boolean expression on a single line
            - Use plain text format (no LaTeX, no markdown, no special formatting)
            - Use uppercase for operators. 
            - Use lowercase for variables: x, y
            - Use parentheses for grouping when needed
            - Start your response with "Expression: " followed by the expression

            Examples of correct format:
            Expression: x AND y
            Expression: (x OR y) AND NOT x
            Expression: NOT (x AND y)
            Expression: NOR(x, y)

            DO NOT use formats like:
            - \\(x \\land y\\)  (LaTeX)
            - `x AND y`  (markdown)
            - x ∧ y  (mathematical symbols)
            """
        return prompt

    def _extract_patterns(self, hypotheses: List[str]) -> str:
        """Extract structural patterns from prior hypotheses to guide diversity"""
        if not hypotheses:
            return ""

        patterns = []
        for h in hypotheses:
            # Count operator usage
            op_counts = {op: h.count(op) for op in self.operators if op in h}
            if op_counts:
                patterns.append(f"uses {', '.join([f'{k} ({v}x)' for k, v in op_counts.items()])}")

            # Check nesting depth
            depth = self._get_expression_depth(h)
            patterns.append(f"nesting depth {depth}")

            # Identify variable combinations
            vars_used = [v for v in self.variables if v in h]
            patterns.append(f"variables: {', '.join(vars_used)}")

        # Return top 2 most common patterns
        from collections import Counter
        top_patterns = [p for p, _ in Counter(patterns).most_common(2)]
        return "; ".join(top_patterns)

    def _get_expression_depth(self, expr: str) -> int:
        """Calculate nesting depth of an expression"""
        max_depth = 0
        current_depth = 0
        for char in expr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth

    def parse_llm_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract Boolean expression."""
        if not isinstance(response, str):
            return None

        response = response.strip()

        # Clean common formatting issues
        def clean_expression(expr: str) -> str:
            # Remove LaTeX delimiters
            expr = re.sub(r'\\[()\[\]]', '', expr)
            expr = re.sub(r'\$+', '', expr)

            # Remove markdown backticks
            expr = expr.strip('`')

            # Convert LaTeX/math symbols to text
            expr = expr.replace('\\land', 'AND')
            expr = expr.replace('\\lor', 'OR')
            expr = expr.replace('\\lnot', 'NOT')
            expr = expr.replace('\\neg', 'NOT')
            expr = expr.replace('\\wedge', 'AND')
            expr = expr.replace('\\vee', 'OR')
            expr = expr.replace('∧', 'AND')
            expr = expr.replace('∨', 'OR')
            expr = expr.replace('¬', 'NOT')
            expr = expr.replace('⊕', 'XOR')

            # Remove \text{} wrappers
            expr = re.sub(r'\\text\{([^}]+)\}', r'\1', expr)

            # Clean up extra spaces and symbols
            expr = re.sub(r'\*+$', '', expr)  # Remove trailing asterisks
            expr = expr.strip('"\'').rstrip('.,;')

            return expr.strip()

        # Look for "Expression:" line
        lines = response.split('\n')
        for line in lines:
            if 'Expression:' in line or 'expression:' in line:
                parts = re.split(r'[Ee]xpression:', line)
                if len(parts) >= 2:
                    expr = clean_expression(parts[1])
                    if expr:
                        return expr

        # Fallback: try first non-empty line with operators
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                cleaned = clean_expression(line)
                if any(op in cleaned.upper() for op in ['AND', 'OR', 'NOT', 'XOR', 'NOR']):
                    return cleaned
                if all(c.isalnum() or c in '() ' for c in cleaned):
                    return cleaned

        return None

    def get_expression_mechanistic_key(self, expr_str: str) -> Optional[Tuple]:
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            return expr.mechanistic_key(**self.mechanistic_opts)
        except:
            return None

    def _in_space(self, expr_str: str) -> bool:
        """Check if expression meets all space constraints (variables, operators, depth, no constants)."""
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)

            if expr.sympy_expr is None:
                return False

            # Check variables
            allowed_symbols = {sp.symbols(v) for v in self.variables}
            if not expr.sympy_expr.free_symbols.issubset(allowed_symbols):
                return False

            # Check operators
            if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                return False

            # Check no constants
            for node in sp.preorder_traversal(expr.sympy_expr):
                if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                    return False

            # Check depth
            if self._get_ast_depth(expr.sympy_expr) > self.max_depth:
                return False

            return True
        except:
            return False

    def _get_ast_depth(self, sympy_expr) -> int:
        """Calculate the true AST depth of an expression."""
        if sympy_expr is None:
            return 0

        # Leaf nodes (variables, constants) have depth 0
        if isinstance(sympy_expr, sp.Symbol):
            return 0
        if isinstance(sympy_expr, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
            return 0

        # For operators, depth is 1 + max depth of children
        if hasattr(sympy_expr, 'args') and len(sympy_expr.args) > 0:
            child_depths = [self._get_ast_depth(arg) for arg in sympy_expr.args]
            return 1 + max(child_depths)

        return 0

    def _check_operators_in_ast(self, sympy_expr, allowed_ops: Set[str]) -> bool:
        if sympy_expr is None:
            return False
        stack = [sympy_expr]
        while stack:
            node = stack.pop()

            # NOR pattern: Not(Or(...))
            if isinstance(node, sp.Not) and isinstance(node.args[0], sp.Or):
                if 'NOR' not in allowed_ops:
                    return False
                # treat as primitive; do NOT traverse inside
                continue

            # primitive checks
            if isinstance(node, sp.Not) and 'NOT' not in allowed_ops:
                return False
            elif isinstance(node, sp.And) and 'AND' not in allowed_ops:
                return False
            elif isinstance(node, sp.Or) and 'OR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Xor) and 'XOR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Implies) and 'IMPLIES' not in allowed_ops:
                return False
            elif isinstance(node, sp.Equivalent) and 'EQUIV' not in allowed_ops:
                return False

            if hasattr(node, 'args'):
                stack.extend(node.args)
        return True

    def validate_expression(self, expr_str: str, observations: List[BooleanObservation]) -> Tuple[bool, Optional[Dict]]:
        """Validate if expression is consistent with observations and meets constraints."""
        try:
            # Now create expression and validate
            expr = BooleanExpression(expr_str, self.variables, self.operators)

            # Check if expression uses only allowed variables
            if expr.sympy_expr is not None:
                allowed_symbols = {sp.symbols(v) for v in self.variables}
                free_symbols = expr.sympy_expr.free_symbols
                if not free_symbols.issubset(allowed_symbols):
                    return False, None

                # Check operators using AST traversal
                if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                    return False, None

                # Disallow constants (True/False) to maintain consistent space
                for node in sp.preorder_traversal(expr.sympy_expr):
                    if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                        return False, None

            # Check expression depth using AST
            if expr.sympy_expr is not None:
                ast_depth = self._get_ast_depth(expr.sympy_expr)
                if ast_depth > self.max_depth:
                    return False, None

            # Check consistency with observations
            for obs in observations:
                if expr.evaluate(obs.inputs) != obs.output:
                    return False, None

            return True, expr.truth_table
        except:
            return False, None

    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error from the error message with more detail."""
        if "Expecting value" in error_message:
            # Extract line and column info if available
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower():
            return "rate_limit"
        elif "timeout" in error_message.lower():
            return "timeout"
        elif "401" in error_message or "unauthorized" in error_message.lower():
            return "auth_error"
        elif "403" in error_message or "forbidden" in error_message.lower():
            return "forbidden_error"
        elif "404" in error_message:
            return "not_found_error"
        elif "429" in error_message:
            return "rate_limit_429"
        elif "500" in error_message or "internal server error" in error_message.lower():
            return "server_error_500"
        elif "502" in error_message or "bad gateway" in error_message.lower():
            return "bad_gateway_502"
        elif "503" in error_message or "service unavailable" in error_message.lower():
            return "service_unavailable_503"
        elif "connection" in error_message.lower():
            return "connection_error"
        elif "JSONDecodeError" in error_message:
            return "json_decode_error"
        else:
            # Try to extract HTTP status code
            match = re.search(r'\b(\d{3})\b', error_message)
            if match:
                return f"http_error_{match.group(1)}"
            return "unknown_error"

    def evaluate_single_observation_set(
            self,
            llm: LLMInterface,
            observation_set: Dict,
            n_queries: int = 10,
            verbose: bool = True,
            max_retries: int = 5
    ) -> Dict:
        """
        Evaluate LLM on a single observation set.

        Returns:
            Dictionary with evaluation results for this observation set
        """
        # Extract observations and ground truths
        observations = [
            BooleanObservation(obs['inputs'], obs['output'])
            for obs in observation_set['observations']
        ]

        ground_truth_expressions = observation_set['ground_truth_expressions']
        gt_truth_tables = []
        gt_mechanistic_keys = []
        for gt in ground_truth_expressions:
            truth_table = {}
            for key_str, value in gt['truth_table'].items():
                key_tuple = ast.literal_eval(key_str)
                truth_table[key_tuple] = value
            gt_truth_tables.append(truth_table)
            # Extract mechanistic key for structural comparison
            gt_mech_key = gt.get('mechanistic_key', None)
            if gt_mech_key:
                # Convert string representation back to tuple if needed (safe parsing)
                gt_mechanistic_keys.append(
                    ast.literal_eval(gt_mech_key) if isinstance(gt_mech_key, str) else gt_mech_key)

        # Track results
        all_hypotheses = []
        valid_hypotheses = []
        unique_mechanistic_keys = set()
        unique_valid_expressions = []
        all_unique_mechanistic_keys = set()
        unique_all_expressions = []
        parse_success_count = 0
        in_space_count = 0

        # Track token usage and costs
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0

        # Track errors
        errors = []  # List of error details
        error_counts = {}  # Count of each error type

        # Track operator distribution in valid expressions
        operator_distribution = defaultdict(int)
        depth_distribution = defaultdict(int)

        for i in range(n_queries):
            prompt = self.create_prompt(observations, all_hypotheses)

            # --- Self-Consistency sampling ---
            sc_k = 5  # number of samples per prompt
            candidates: List[str] = []
            last_error = None
            for _ in range(sc_k):
                hypothesis_str = None
                query_error = None
                for attempt in range(max_retries):
                    if hasattr(llm, 'query_with_usage'):
                        result = llm.query_with_usage(prompt)
                        response = result['response']
                        total_prompt_tokens += result['usage'].get('prompt_tokens', 0)
                        total_completion_tokens += result['usage'].get('completion_tokens', 0)
                        total_tokens += result['usage'].get('total_tokens', 0)
                        total_cost += result.get('cost', 0.0)
                    else:
                        response = llm.query(prompt)

                    if isinstance(response, str) and response.startswith("Error querying"):
                        query_error = {
                            'query_index': i,
                            'attempt': attempt + 1,
                            'error_message': response,
                            'error_type': self._classify_error(response)
                        }
                        error_type = query_error['error_type']
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                        continue

                    hypothesis_str = self.parse_llm_response(response)
                    if hypothesis_str:
                        break
                if hypothesis_str:
                    candidates.append(hypothesis_str)
                else:
                    last_error = query_error

            # If no candidate parsed at all
            if not candidates:
                if last_error:
                    errors.append(last_error)
                    error_detail = f"[ERROR after {max_retries} attempts] Type: {last_error['error_type']} | {last_error['error_message']}"
                    all_hypotheses.append(error_detail)
                continue

            # Score candidates: prefer valid & structurally new; else minimal mismatch
            best = None
            best_key = None
            best_mismatch = 10**9
            best_depth = 10**9
            best_ops = 10**9

            for cand in candidates:
                # score candidates only; do NOT update global metrics here
                in_space = self._in_space(cand)
                is_valid, _ = self.validate_expression(cand, observations)

                if is_valid:
                    key = self.get_expression_mechanistic_key(cand)
                    d, oc = _expr_depth_and_ops(cand, self.variables, self.operators)
                    novelty_bonus = 0 if (key is not None and key in unique_mechanistic_keys) else -1
                    rank = (novelty_bonus, d, oc)
                    current_rank = (0 if (best_key is not None and best_key in unique_mechanistic_keys) else -1, best_depth, best_ops)
                    if best is None or rank < current_rank:
                        best = cand
                        best_key = key
                        best_depth, best_ops = d, oc
                else:
                    mm = _boolean_mismatch_score(cand, observations, self.variables, self.operators)
                    d, oc = _expr_depth_and_ops(cand, self.variables, self.operators)
                    if best is None and mm < best_mismatch:
                        best = cand
                        best_mismatch = mm
                        best_depth, best_ops = d, oc

            if best is None:
                # Fallback: record the first candidate
                best = candidates[0]

            # Record best candidate
            hypothesis_str = best

            # Validate and update counters
            if hypothesis_str:
                # record only the accepted expression for this query
                all_hypotheses.append(hypothesis_str)
                parse_success_count += 1
                # Only count novelty for in-space expressions
                if self._in_space(hypothesis_str):
                    in_space_count += 1
                    all_key = self.get_expression_mechanistic_key(hypothesis_str)
                    if all_key and all_key not in all_unique_mechanistic_keys:
                        all_unique_mechanistic_keys.add(all_key)
                        unique_all_expressions.append(hypothesis_str)

                is_valid, _ = self.validate_expression(hypothesis_str, observations)
                if is_valid:
                    valid_hypotheses.append(hypothesis_str)
                    mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                    if mech_key and mech_key not in unique_mechanistic_keys:
                        unique_mechanistic_keys.add(mech_key)
                        unique_valid_expressions.append(hypothesis_str)
                    # Track operator and depth
                    expr_upper = hypothesis_str.upper()
                    for op in ['AND', 'OR', 'NOT', 'XOR', 'NOR']:
                        if op in expr_upper:
                            operator_distribution[op] += 1
                    depth = self._get_expression_depth(hypothesis_str)
                    depth_distribution[depth] += 1

        # Calculate metrics
        denom = max(1, n_queries)
        parse_success_rate = min(1.0, parse_success_count / denom)
        in_space_rate = min(1.0, in_space_count / denom)
        valid_rate = min(1.0, len(valid_hypotheses) / denom)
        novelty_rate = min(1.0, len(unique_all_expressions) / denom)
        recovery_rate = 0

        # Check recovery against ground truths using mechanistic keys (structural matching)
        recovered_gts = set()
        recovered_gt_keys = set()

        # Collect mechanistic keys of generated expressions
        generated_mech_keys = set()
        for expr in unique_valid_expressions:
            mech_key = self.get_expression_mechanistic_key(expr)
            if mech_key:
                generated_mech_keys.add(mech_key)

        # Check which ground truths were recovered (structurally)
        for j, gt_key in enumerate(gt_mechanistic_keys):
            if gt_key in generated_mech_keys:
                recovered_gts.add(j)
                recovered_gt_keys.add(gt_key)

        # (Optional extension) Early-stop: For future, could add early stopping if all GTs recovered here.
        recovery_rate = len(recovered_gts) / len(gt_mechanistic_keys) if gt_mechanistic_keys else 0

        return {
            'observation_set_id': observation_set['observation_set_id'],
            'n_observations': observation_set['n_observations'],
            'n_ground_truths': len(ground_truth_expressions),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_expressions),  # Unique among valid hypotheses
            'n_unique_all': len(unique_all_expressions),  # Unique among in-space hypotheses
            'n_recovered_gts': len(recovered_gts),
            'parse_success_rate': parse_success_rate,  # Diagnostic: how many parsed
            'in_space_rate': in_space_rate,  # Diagnostic: how many met space constraints
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,  # Now using unique in-space / n_queries
            'recovery_rate': recovery_rate,
            'all_hypotheses': all_hypotheses,
            'valid_hypotheses': valid_hypotheses,
            'unique_valid_expressions': unique_valid_expressions,
            'unique_all_expressions': unique_all_expressions,
            # Token usage and cost tracking
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            # Error tracking
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            },
            # Metrics: operator and depth distribution
            'operator_distribution': dict(operator_distribution),
            'depth_distribution': dict(depth_distribution)
        }

    def run_benchmark(
            self,
            llm: LLMInterface,
            n_samples: int = 10,
            n_queries_per_sample: Optional[int] = None,
            query_multiplier: float = 2.0,
            seed: Optional[int] = None,
            verbose: bool = True,
            checkpoint_dir: str = "checkpoints",
            run_id: Optional[str] = None
    ) -> Dict:
        """
        Run the benchmark with sampling.

        Args:
            llm: LLM interface to use
            n_samples: Number of observation sets to sample
            n_queries_per_sample: Fixed number of queries per observation set (if None, uses query_multiplier)
            query_multiplier: Multiplier for n_gt to determine queries (default 2.0 means 2x ground truths)
            seed: Random seed
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary with complete benchmark results
        """
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Generate run ID and safe filename
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize LLM name for filename
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_{safe_llm_name}_{run_id}.json"

        print(f"\nRunning refined Boolean benchmark")
        print(f"LLM: {llm.get_name()}")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Checkpoint file: {checkpoint_file}")
        print("-" * 50)

        # Sample observation sets
        sampled_sets = self.sample_observation_sets(n_samples, seed)

        # Initialize results tracking
        all_results = []
        valid_rates = []
        novelty_rates = []
        recovery_rates = []

        # Initialize token and cost tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0

        # Initialize error tracking
        all_errors = []
        total_error_counts = {}

        # Track aggregated operator and depth distributions
        total_operator_distribution = defaultdict(int)
        total_depth_distribution = defaultdict(int)

        # Load existing checkpoint if it exists
        start_idx = 0
        if checkpoint_file.exists():
            print(f"Using Recovered checkpoint file: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                all_results = checkpoint_data['results']
                start_idx = len(all_results)
                print(f"Resuming from checkpoint: {start_idx}/{n_samples} completed")

                # Recalculate rates from checkpoint
                for result in all_results:
                    valid_rates.append(result['valid_rate'])
                    novelty_rates.append(result['novelty_rate'])
                    recovery_rates.append(result['recovery_rate'])

                    # Restore aggregated distributions
                    for op, count in result.get('operator_distribution', {}).items():
                        total_operator_distribution[op] += count
                    for depth, count in result.get('depth_distribution', {}).items():
                        total_depth_distribution[depth] += count

                # RESTORE TOKEN AND COST TOTALS
                if 'total_token_usage' in checkpoint_data:
                    total_prompt_tokens = checkpoint_data['total_token_usage'].get('prompt_tokens', 0)
                    total_completion_tokens = checkpoint_data['total_token_usage'].get('completion_tokens', 0)
                    total_tokens = checkpoint_data['total_token_usage'].get('total_tokens', 0)
                if 'total_cost' in checkpoint_data:
                    total_cost = checkpoint_data['total_cost']

        # Process each sampled observation set
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]

            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                print(f"  Observation set ID: {obs_set['observation_set_id']}")
                print(f"  Number of observations: {obs_set['n_observations']}")
                print(f"  Number of ground truths: {obs_set['n_compatible_expressions']}")

            try:
                # Determine number of queries for this observation set
                if n_queries_per_sample is not None:
                    # Use fixed number
                    n_queries = n_queries_per_sample
                else:
                    # Adaptive: use multiplier of ground truths
                    n_gt = obs_set['n_compatible_expressions']
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Using {n_queries} queries ({query_multiplier}x {n_gt} ground truths)")

                # Evaluate on this observation set
                result = self.evaluate_single_observation_set(
                    llm, obs_set, n_queries, verbose=False
                )

                all_results.append(result)
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])

                # Aggregate distributions
                for op, count in result.get('operator_distribution', {}).items():
                    total_operator_distribution[op] += count
                for depth, count in result.get('depth_distribution', {}).items():
                    total_depth_distribution[depth] += count

                # Aggregate token usage and costs
                if 'token_usage' in result:
                    total_prompt_tokens += result['token_usage']['prompt_tokens']
                    total_completion_tokens += result['token_usage']['completion_tokens']
                    total_tokens += result['token_usage']['total_tokens']
                if 'cost' in result:
                    total_cost += result['cost']

                # Aggregate errors
                if 'errors' in result and result['errors']:
                    all_errors.extend(result['errors'])
                    # Update error type counts
                    if 'error_summary' in result:
                        for error_type, count in result['error_summary']['error_types'].items():
                            total_error_counts[error_type] = total_error_counts.get(error_type, 0) + count

                if verbose:
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")

                # Save checkpoint after each sample
                checkpoint_data = {
                    'run_id': run_id,
                    'llm_name': llm.get_name(),
                    'n_samples': n_samples,
                    'n_queries_per_sample': n_queries_per_sample,
                    'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results,
                    'total_token_usage': {
                        'prompt_tokens': total_prompt_tokens,
                        'completion_tokens': total_completion_tokens,
                        'total_tokens': total_tokens
                    },
                    'total_cost': total_cost,
                    'total_errors': len(all_errors),
                    'error_types': total_error_counts,
                    'total_operator_distribution': dict(total_operator_distribution),
                    'total_depth_distribution': dict(total_depth_distribution)
                }

                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

            except Exception as e:
                print(f"  Error processing sample {idx + 1}: {str(e)}")
                traceback.print_exc()
                continue

        # Calculate statistics
        def calculate_stats(rates):
            if not rates:
                return {'mean': 0, 'std': 0, 'var': 0, 'min': 0, 'max': 0}
            return {
                'mean': np.mean(rates),
                'std': np.std(rates),
                'var': np.var(rates),
                'min': np.min(rates),
                'max': np.max(rates)
            }

        # Calculate p-values (one-sample t-test against null hypothesis of 0)
        def calculate_p_value(rates):
            if not rates or len(rates) < 2:
                return None
            t_stat, p_val = stats.ttest_1samp(rates, 0)
            return p_val

        # Compile final results
        final_results = {
            'run_id': run_id,
            'llm_name': llm.get_name(),
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'valid_rate': {
                    **calculate_stats(valid_rates),
                    'p_value': calculate_p_value(valid_rates)
                },
                'novelty_rate': {
                    **calculate_stats(novelty_rates),
                    'p_value': calculate_p_value(novelty_rates)
                },
                'recovery_rate': {
                    **calculate_stats(recovery_rates),
                    'p_value': calculate_p_value(recovery_rates)
                }
            },
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens,
                'avg_tokens_per_sample': total_tokens / len(all_results) if all_results else 0,
                'avg_tokens_per_query': total_tokens / (
                            len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'cost': {
                'total_cost': total_cost,
                'avg_cost_per_sample': total_cost / len(all_results) if all_results else 0,
                'avg_cost_per_query': total_cost / (
                            len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'error_summary': {
                'total_errors': len(all_errors),
                'error_types': total_error_counts,
                'error_rate': len(all_errors) / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            # Aggregated metrics
            'operator_usage': {
                'distribution': dict(total_operator_distribution),
                'total_operators': sum(total_operator_distribution.values()),
                'percentage': {op: count / sum(total_operator_distribution.values()) * 100
                               for op, count in total_operator_distribution.items()
                               if sum(total_operator_distribution.values()) > 0}
            },
            'depth_usage': {
                'distribution': dict(total_depth_distribution),
                'avg_depth': sum(int(d) * c for d, c in total_depth_distribution.items()) /
                             sum(total_depth_distribution.values()) if sum(total_depth_distribution.values()) > 0 else 0
            },
            'per_sample_results': all_results
        }

        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 50)
        print(f"Samples evaluated: {len(all_results)}/{n_samples}")

        for metric_name, metric_key in [('Valid Rate', 'valid_rate'),
                                        ('Novelty Rate', 'novelty_rate'),
                                        ('Recovery Rate', 'recovery_rate')]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance: {stats_dict['var']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value: {stats_dict['p_value']:.4f}")

        # Print operator usage summary
        print("\nOperator Usage:")
        if final_results['operator_usage']['total_operators'] > 0:
            for op, pct in final_results['operator_usage']['percentage'].items():
                print(f"  {op}: {pct:.1f}% ({final_results['operator_usage']['distribution'][op]})")
        else:
            print("  No valid expressions to analyze")

        # Print depth usage summary
        print("\nExpression Depth:")
        if sum(final_results['depth_usage']['distribution'].values()) > 0:
            print(f"  Average depth: {final_results['depth_usage']['avg_depth']:.1f}")
            print("  Depth distribution:")
            for depth in sorted(final_results['depth_usage']['distribution'].keys()):
                print(f"    Depth {depth}: {final_results['depth_usage']['distribution'][depth]} expressions")
        else:
            print("  No valid expressions to analyze")

        # Print token usage and cost summary
        print(f"\nToken Usage:")
        print(f"  Total tokens: {final_results['token_usage']['total_tokens']:,}")
        print(f"  Prompt tokens: {final_results['token_usage']['prompt_tokens']:,}")
        print(f"  Completion tokens: {final_results['token_usage']['completion_tokens']:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")

        print(f"\nCost:")
        print(f"  Total cost: ${final_results['cost']['total_cost']:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")

        # Print error summary if there were errors
        if final_results['error_summary']['total_errors'] > 0:
            print(f"\nErrors:")
            print(f"  Total errors: {final_results['error_summary']['total_errors']}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if final_results['error_summary']['error_types']:
                print(f"  Error types:")
                for error_type, count in final_results['error_summary']['error_types'].items():
                    print(f"    - {error_type}: {count}")

        print("=" * 50)

        return final_results


def setup_llm(llm_type: str, **kwargs) -> LLMInterface:
    """Set up the LLM interface based on type."""

    if llm_type == "openai":
        api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")

        return OpenAILLM(
            model=kwargs.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    elif llm_type == "anthropic":
        api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required")

        return AnthropicLLM(
            model=kwargs.get('model', 'claude-3-opus-20240229'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    elif llm_type == "openrouter":
        api_key = kwargs.get('api_key') or os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key required")

        return OpenRouterLLM(
            model=kwargs.get('model', 'anthropic/claude-3.5-sonnet'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run refined Boolean discovery benchmark with sampling")
    parser.add_argument("--dataset", required=True, help="Path to complete Boolean dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of observation sets to sample")
    parser.add_argument("--n-queries", type=int, default=None,
                        help="Fixed number of queries per observation set (if not set, uses adaptive)")
    parser.add_argument("--query-multiplier", type=float, default=1.0,
                        help="Multiplier for adaptive queries (n_queries = n_gt * multiplier)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default=None, help="Output file path for results")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    llm_type = config.get('llm', {}).get('type', 'openrouter')

    model = config.get('llm', {}).get('models', {}).get(llm_type)
    if not model:
        default_models = {
            'openrouter': 'openai/gpt-3.5-turbo',
            'openai': 'gpt-4',
            'anthropic': 'claude-3-opus-20240229'
        }
        model = default_models.get(llm_type)

    api_key = config.get('llm', {}).get('api_keys', {}).get(llm_type)
    if not api_key:
        env_vars = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY'
        }
        if llm_type in env_vars:
            api_key = os.environ.get(env_vars[llm_type])

    temperature = config.get('llm', {}).get('temperature', 0.7)
    checkpoint_dir = config.get('benchmark', {}).get('checkpoint', 'checkpoints')
    run_id = config.get('benchmark', {}).get('run_id', None)
    verbose = config.get('benchmark', {}).get('verbose', True)

    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    # Generate output filename if not specified
    if args.output is None:
        dataset_name = Path(args.dataset).stem
        model_name = Path(model).stem if model else llm_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"results/{dataset_name}_{model_name}_{timestamp}.json"
    else:
        output = args.output

    # Initialize benchmark
    benchmark = BooleanBenchmarkRefined(args.dataset)

    # Print configuration
    print("\n" + "=" * 60)
    print("REFINED BOOLEAN BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"LLM Type: {llm_type}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Samples: {args.n_samples}")
    print(f"Queries per sample: {args.n_queries}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output}")
    print("=" * 60)

    # Set up LLM
    llm = setup_llm(
        llm_type,
        model=model,
        api_key=api_key,
        temperature=temperature
    )

    # Run benchmark
    results = benchmark.run_benchmark(
        llm=llm,
        n_samples=args.n_samples,
        n_queries_per_sample=args.n_queries,
        query_multiplier=args.query_multiplier,
        seed=args.seed,
        verbose=verbose,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id
    )

    # Save final results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal results saved to: {output}")


if __name__ == "__main__":
    main()