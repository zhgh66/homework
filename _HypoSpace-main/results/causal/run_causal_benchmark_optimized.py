 # JSON helper to serialize NumPy types
def _convert_numpy(obj):
    import numpy as _np
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from textwrap import dedent
from collections import Counter
from scipy import stats
import traceback

from modules.models import CausalGraph
from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM
from generate_causal_dataset import PerturbationObservation, CausalDatasetGenerator


class CausalBenchmarkEnhanced:
    """Enh    增强版因果因果图发现的LLM基准测试类，包含动态提示优化和智能查询策略
    """

    def __init__(self, complete_dataset_path: Optional[str] = None,
                 n_observations_filter: Optional[List[int]] = None,
                 gt_filter: Optional[Tuple] = None):
        # 保持原初始化逻辑不变
        self.n_observations_filter = n_observations_filter
        self.gt_filter = gt_filter
        self.filtered_observation_sets = []
        self.excluded_observation_sets = []  # 用于潜在的回填

        if complete_dataset_path:
            with open(complete_dataset_path, 'r') as f:
                self.complete_dataset = json.load(f)

            # 提取元数据
            self.metadata = self.complete_dataset.get('metadata', {})
            self.nodes = self.metadata.get('nodes', [])
            self.max_edges = self.metadata.get('max_edges', None)

            # 扁平化所有数据集以便采样
            self.all_observation_sets = []
            if 'datasets_by_n_observations' in self.complete_dataset:
                for n_obs, datasets in self.complete_dataset['datasets_by_n_observations'].items():
                    self.all_observation_sets.extend(datasets)
            elif 'datasets' in self.complete_dataset:
                self.all_observation_sets = self.complete_dataset['datasets']
            elif 'sampled_datasets' in self.complete_dataset:
                self.all_observation_sets = self.complete_dataset['sampled_datasets']

            # 应用两阶段过滤
            stage1_filtered = self.all_observation_sets

            # 第一阶段：应用观测数过滤
            if n_observations_filter:
                stage1_filtered = [
                    obs_set for obs_set in self.all_observation_sets
                    if obs_set.get('n_observations') in n_observations_filter
                ]
                print(f"Stage 1 filter: kept {len(stage1_filtered)} sets matching the number-of-observations criteria")

            # 第二阶段：应用GT过滤
            if gt_filter:
                if gt_filter[1] is not None:  # 范围过滤 (min, max)
                    min_gt, max_gt = gt_filter
                    self.filtered_observation_sets = [
                        obs_set for obs_set in stage1_filtered
                        if min_gt <= obs_set.get('n_compatible_graphs', 0) <= max_gt
                    ]
                    print(f"Stage 2 filter: kept {len(self.filtered_observation_sets)} sets within the GT-count range")
                else:  # 特定值过滤
                    allowed_values = gt_filter[0] if isinstance(gt_filter[0], list) else []
                    self.filtered_observation_sets = [
                        obs_set for obs_set in stage1_filtered
                        if obs_set.get('n_compatible_graphs', 0) in allowed_values
                    ]
                    print(f"Stage 2 filter: kept {len(self.filtered_observation_sets)} sets with allowed GT counts")

                # 保留被排除的集合用于潜在回填
                self.excluded_observation_sets = [
                    obs_set for obs_set in self.all_observation_sets
                    if obs_set not in self.filtered_observation_sets
                ]
            else:
                self.filtered_observation_sets = stage1_filtered
                self.excluded_observation_sets = []

            # 从GT中推断最大边数（如果未指定）
            if self.max_edges is None and self.all_observation_sets:
                max_edges_in_gts = 0
                for obs_set in self.all_observation_sets:
                    for gt in obs_set.get('ground_truth_graphs', []):
                        num_edges = len(gt.get('edges', []))
                        max_edges_in_gts = max(max_edges_in_gts, num_edges)
                self.max_edges = max_edges_in_gts
                print(f"Inferred max_edges from ground truths = {self.max_edges}")

            print(f"Loaded full dataset with {len(self.all_observation_sets)} observation sets")
            if n_observations_filter or gt_filter:
                print(f"After filtering: {len(self.filtered_observation_sets)} sets meet the criteria")
                if self.excluded_observation_sets:
                    print(f"  ({len(self.excluded_observation_sets)} sets available for backfilling)")
            print(f"Nodes: {', '.join(self.nodes)}")
            print(f"Max edges in hypothesis space: {self.max_edges if self.max_edges is not None else 'unbounded'}")
        else:
            self.complete_dataset = None
            self.all_observation_sets = []
            self.filtered_observation_sets = []
            print("Initialized empty benchmark – dataset will be generated on demand")

    def sample_observation_sets(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        # 保持原采样逻辑不变
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 根据过滤条件确定主池
        if self.n_observations_filter or self.gt_filter:
            primary_pool = self.filtered_observation_sets
            backup_pool = self.excluded_observation_sets
        else:
            primary_pool = self.all_observation_sets
            backup_pool = []

        sampled = []

        # 首先从主池采样（符合过滤条件的数据集）
        n_primary = len(primary_pool)
        if n_primary > 0:
            n_from_primary = min(n_samples, n_primary)
            sampled_primary = random.sample(primary_pool, n_from_primary)
            sampled.extend(sampled_primary)

            # 标记这些集合符合过滤条件
            for obs_set in sampled_primary:
                obs_set['meets_filter_criteria'] = True

        # 如果需要更多样本，从备份池回填
        n_still_needed = n_samples - len(sampled)
        if n_still_needed > 0 and backup_pool:
            n_backup = len(backup_pool)
            n_from_backup = min(n_still_needed, n_backup)

            if n_from_backup > 0:
                print(f"\nBackfilling: only {n_primary} datasets met the filter criteria")
                print(f"  Randomly adding {n_from_backup} datasets from outside the filter range")

                sampled_backup = random.sample(backup_pool, n_from_backup)

                # 标记这些集合为回填
                for obs_set in sampled_backup:
                    obs_set['meets_filter_criteria'] = False
                    obs_set['backfilled'] = True

                sampled.extend(sampled_backup)

        # 最终检查是否仍不足够
        if len(sampled) < n_samples:
            total_available = len(primary_pool) + len(backup_pool)
            print(f"\nWarning: requested {n_samples} samples but only {total_available} available")
            print(f"  Returning {len(sampled)} datasets")

        return sampled

    def create_prompt(self, observations: List[Dict], prior_hypotheses: List[CausalGraph],
                      ground_truths: Optional[List[CausalGraph]] = None) -> str:
        """
        创建优化的提示词，增加了结构化指导和动态约束

        Args:
            observations: 观测数据列表
            prior_hypotheses: 之前生成的假设
            ground_truths: 真实因果图（用于提示优化）
        """
        nodes_str = ", ".join(self.nodes)
        obs_block = "\n".join(obs["string"] for obs in observations)

        # 构建历史假设块（限制显示最近3个以避免提示过长）
        prior_block = "None"
        if prior_hypotheses:
            prior_lines = []
            for h in prior_hypotheses[-3:]:  # 只显示最近3个假设
                edges = [f"{s}->{d}" for s, d in h.edges]
                if edges:
                    prior_lines.append("Graph: " + ", ".join(edges))
                else:
                    prior_lines.append("Graph: No edges")
            prior_block = "\n".join(prior_lines)

        # 增加边数约束信息
        constraint_info = ""
        if self.max_edges is not None:
            constraint_info = f"\nConstraint: the graph may contain at most {self.max_edges} edges"

        # 动态提示增强：基于未发现的GT提供指导
        recovery_guide = ""
        if ground_truths and prior_hypotheses:
            # 找出尚未被发现的GT
            covered_gt = set()
            for hyp in prior_hypotheses:
                for gt in ground_truths:
                    if hyp.get_hash() == gt.get_hash():
                        covered_gt.add(gt.get_hash())

            uncovered_gt = [gt for gt in ground_truths if gt.get_hash() not in covered_gt]
            if uncovered_gt and len(prior_hypotheses) > 0:
                # 提取未覆盖GT中的边作为提示
                unique_edges = set()
                for gt in uncovered_gt:
                    for edge in gt.edges:
                        unique_edges.add(f"{edge[0]}->{edge[1]}")

                recovery_guide = f"\nTip: Consider including these potential edges to cover more possible causal relationships: {', '.join(unique_edges)}"

        prompt = f"""
        You must infer the causal system structure from perturbation experiments.

        Semantics:
        - When a node is perturbed, that node has value 0.
        - A node has value 1 iff it is a descendant of the perturbed node in the DAG (reachable along directed edges).
        - All other nodes have value 0.

        Nodes: {nodes_str}{constraint_info}

        Observations:
        {obs_block}

        Prior attempts (avoid repeating exact edge sets):
        {prior_block}

        {recovery_guide}

        Task:
        Output a directed acyclic graph (DAG) that matches all observations.

        Diversity note:
        - A "diverse" graph means its edge set differs from any prior attempt.
        - Prefer exploring diverse valid graphs to cover the solution space.

        Output format rules:
        1) Use only the listed nodes; no self-loops; no cycles.
        2) Respond on a single line only:
        - With edges: Graph: A->B, B->C
        - No edges:  Graph: No edges
        """
        return dedent(prompt).strip()

    def parse_llm_response(self, response: str) -> Optional[CausalGraph]:
        # 保持原解析逻辑不变
        if not isinstance(response, str):
            return None

        # 去除代码块和空白
        s = response.replace("```", "").strip()

        # 查找"Graph:"行
        m = re.search(r'(?i)\bgraph\s*:\s*(.+)$', s, flags=re.MULTILINE)
        line = m.group(1).strip() if m else (s.splitlines()[0].strip() if s.splitlines() else "")
        if not line:
            return None

        # 清理行内容
        if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            line = line[1:-1].strip()
        line = (line
                .replace("→", "->")
                .replace("-->", "->")
                .replace("=>", "->")
                .rstrip(" .;"))

        # 处理"No edges"情况
        if re.search(r'\b(no\s+edges?|empty|none|null)\b', line, re.I):
            return CausalGraph(nodes=self.nodes, edges=[])

        # 解析边
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            return None

        edges = []
        for part in parts:
            m = re.fullmatch(r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)', part)
            if not m:
                return None
            u, v = m.group(1), m.group(2)
            if u not in self.nodes or v not in self.nodes or u == v:
                return None
            edges.append((u, v))

        # 去重
        edges = list(dict.fromkeys(edges))

        # 检查最大边数约束
        if self.max_edges is not None and len(edges) > self.max_edges:
            return None  # 边数过多

        # 验证DAG
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(G):
            return None

        return CausalGraph(nodes=self.nodes, edges=edges)

    def validate_hypothesis(self, hypothesis: CausalGraph, observations: List[Dict]) -> bool:
        # 保持原验证逻辑不变
        for obs_dict in observations:
            perturbed_node = obs_dict['perturbed_node']
            expected_effects = obs_dict['effects']

            # 获取假设预测的结果
            hypothesis_obs = CausalDatasetGenerator.get_perturbation_effects(hypothesis, perturbed_node)

            # 检查是否匹配预期结果
            if hypothesis_obs.effects != expected_effects:
                return False

        return True

    def _classify_error(self, error_message: str) -> str:
        # 保持原错误分类逻辑不变
        if "Expecting value" in error_message:
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower() or "rate_limit" in error_message.lower():
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
            max_retries: int = 5,
            dynamic_query: bool = True  # 新增：动态查询策略开关
    ) -> Dict:
        """
        评估LLM在单个观测集合上的表现，增加动态查询策略

        Args:
            dynamic_query: 是否使用动态查询策略（当覆盖所有GT时提前停止）
        """
        # 提取观测和真实图
        observations = observation_set['observations']
        ground_truth_graphs = [
            CausalGraph.from_dict(g) for g in observation_set['ground_truth_graphs']
        ]

        # 获取GT哈希用于检查恢复情况
        gt_hashes = {g.get_hash() for g in ground_truth_graphs}
        total_gt = len(gt_hashes)

        # 跟踪结果
        all_hypotheses = []
        valid_hypotheses = []
        unique_hashes = set()
        unique_valid_graphs = []
        all_unique_hashes = set()
        unique_all_graphs = []
        parse_success_count = 0

        # 令牌和成本跟踪
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0

        # 错误跟踪
        errors = []
        error_counts = {}

        # 动态查询策略：已恢复的GT计数
        recovered_gt_count = 0

        for i in range(n_queries):
            # 动态查询策略：如果已发现所有GT且启用了动态策略，则提前停止
            if dynamic_query and recovered_gt_count == total_gt and total_gt > 0:
                print(f"  All {total_gt} ground truths recovered – early stop")
                break

            # 创建提示词，传入GT用于优化提示
            prompt = self.create_prompt(observations, all_hypotheses, ground_truth_graphs)

            # 尝试获取有效响应
            hypothesis = None
            query_error = None

            for attempt in range(max_retries):
                try:
                    # 使用带使用量跟踪的查询方法
                    if hasattr(llm, 'query_with_usage'):
                        result = llm.query_with_usage(prompt)
                        response = result['response']

                        # 跟踪使用量
                        usage = result.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0)
                        total_completion_tokens += usage.get('completion_tokens', 0)
                        total_tokens += usage.get('total_tokens', 0)
                        total_cost += result.get('cost', 0.0)
                    else:
                        response = llm.query(prompt)

                    # 检查响应是否为错误
                    if response and response.startswith("Error querying"):
                        query_error = {
                            'query_index': i,
                            'attempt': attempt + 1,
                            'error_message': response,
                            'error_type': self._classify_error(response)
                        }
                        error_type = query_error['error_type']
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                        continue

                    # 解析响应
                    hypothesis = self.parse_llm_response(response)
                    if hypothesis:
                        parse_success_count += 1
                        break

                except Exception as e:
                    query_error = {
                        'query_index': i,
                        'attempt': attempt + 1,
                        'error_message': str(e),
                        'error_type': self._classify_error(str(e))
                    }
                    if verbose:
                        print(f"  ⚠ Query {i + 1} exception: {str(e)[:100]}")

            # 如果所有尝试都失败，记录错误
            if not hypothesis and query_error:
                errors.append(query_error)

            if hypothesis:
                all_hypotheses.append(hypothesis)

                # 检查在所有假设中的唯一性（用于新颖性计算）
                all_h_hash = hypothesis.get_hash()
                if all_h_hash not in all_unique_hashes:
                    all_unique_hashes.add(all_h_hash)
                    unique_all_graphs.append(hypothesis)

                # 验证假设
                is_valid = self.validate_hypothesis(hypothesis, observations)

                if is_valid:
                    valid_hypotheses.append(hypothesis)

                    # 检查在有效假设中的唯一性
                    h_hash = hypothesis.get_hash()
                    if h_hash not in unique_hashes:
                        unique_hashes.add(h_hash)
                        unique_valid_graphs.append(hypothesis)

                        # 更新已恢复的GT计数
                        if h_hash in gt_hashes:
                            recovered_gt_count = len([g for g in unique_valid_graphs
                                                      if g.get_hash() in gt_hashes])

        # 计算指标
        actual_queries = len(all_hypotheses) + len(errors)  # 实际执行的查询数（包括错误）
        valid_rate = len(valid_hypotheses) / actual_queries if actual_queries > 0 else 0
        novelty_rate = len(unique_all_graphs) / actual_queries if actual_queries > 0 else 0
        parse_success_rate = parse_success_count / actual_queries if actual_queries > 0 else 0

        # 检查相对于真实图的恢复情况
        recovered_gts = set()
        for graph in unique_valid_graphs:
            if graph.get_hash() in gt_hashes:
                recovered_gts.add(graph.get_hash())

        recovery_rate = len(recovered_gts) / len(gt_hashes) if gt_hashes else 0

        return {
            'observation_set_id': observation_set.get('observation_set_id', 'unknown'),
            'n_observations': len(observations),
            'n_ground_truths': len(ground_truth_graphs),
            'n_queries': actual_queries,  # 返回实际执行的查询数
            'n_queries_planned': n_queries,  # 计划的查询数
            'n_queries_saved': n_queries - actual_queries,  # 动态策略节省的查询数
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_graphs),
            'n_unique_all': len(unique_all_graphs),
            'n_recovered_gts': len(recovered_gts),
            'parse_success_count': parse_success_count,
            'parse_success_rate': parse_success_rate,
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,
            'recovery_rate': recovery_rate,
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            },
            'all_hypotheses': [h.to_dict() for h in all_hypotheses],
            'valid_hypotheses': [h.to_dict() for h in valid_hypotheses],
            'unique_graphs': [g.to_dict() for g in unique_valid_graphs]
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
            max_retries: int = 3,
            dynamic_query: bool = True  # 新增：动态查询策略开关
    ) -> Dict:
        """
        运行增强版基准测试，增加动态查询策略和更详细的结果分析
        """
        # 创建检查点目录
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # 生成运行ID和检查点文件
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_causal_enhanced_{safe_llm_name}_{run_id}.json"

        print(f"\nRunning Enhanced Causal Benchmark")
        print(f"LLM: {llm.get_name()}")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Dynamic query strategy: {'Enabled' if dynamic_query else 'Disabled'}")
        print(f"Max retries: {max_retries}")
        print(f"Checkpoint file: {checkpoint_file}")
        print("-" * 50)

        # 采样观测集合
        sampled_sets = self.sample_observation_sets(n_samples, seed)

        # 初始化结果跟踪
        all_results = []
        valid_rates = []
        novelty_rates = []
        recovery_rates = []
        parse_success_rates = []
        query_savings = []  # 新增：跟踪节省的查询数

        # 令牌和成本跟踪
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0

        # 错误跟踪
        all_errors = []
        total_error_counts = {}

        # 加载已有的检查点（如果存在）
        start_idx = 0
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    all_results = checkpoint_data.get('results', [])
                    start_idx = len(all_results)

                    # 恢复令牌/成本数据
                    total_prompt_tokens = checkpoint_data.get('total_prompt_tokens', 0)
                    total_completion_tokens = checkpoint_data.get('total_completion_tokens', 0)
                    total_tokens = checkpoint_data.get('total_tokens', 0)
                    total_cost = checkpoint_data.get('total_cost', 0.0)

                    # 恢复错误数据
                    all_errors = checkpoint_data.get('all_errors', [])
                    total_error_counts = checkpoint_data.get('total_error_counts', {})

                    # 恢复查询节省数据
                    query_savings = checkpoint_data.get('query_savings', [])

                    print(f"从检查点恢复: {start_idx}/{n_samples} 已完成")

                    # 从检查点重新计算比率
                    for result in all_results:
                        valid_rates.append(result['valid_rate'])
                        novelty_rates.append(result['novelty_rate'])
                        recovery_rates.append(result['recovery_rate'])
                        parse_success_rates.append(result.get('parse_success_rate', 1.0))
            except Exception as e:
                print(f"警告: 加载检查点失败: {e}")
                print("从头开始...")

        # 处理每个采样的观测集合
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]

            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                print(f"  Observation set ID: {obs_set.get('observation_set_id', 'unknown')}")
                print(f"  Number of observations: {len(obs_set['observations'])}")
                print(f"  Number of ground truths: {obs_set['n_compatible_graphs']}")

            try:
                # 确定查询数量
                if n_queries_per_sample is not None:
                    n_queries = n_queries_per_sample
                else:
                    n_gt = obs_set['n_compatible_graphs']
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Using {n_queries} queries ({query_multiplier}x {n_gt} ground truths)")

                # 评估
                result = self.evaluate_single_observation_set(
                    llm, obs_set, n_queries, verbose=False,
                    max_retries=max_retries, dynamic_query=dynamic_query
                )

                all_results.append(result)
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                parse_success_rates.append(result['parse_success_rate'])
                query_savings.append(result['n_queries_saved'])  # 跟踪节省的查询数

                # 汇总令牌使用量和成本
                if 'token_usage' in result:
                    total_prompt_tokens += result['token_usage']['prompt_tokens']
                    total_completion_tokens += result['token_usage']['completion_tokens']
                    total_tokens += result['token_usage']['total_tokens']
                if 'cost' in result:
                    total_cost += result['cost']

                # 汇总错误
                if 'errors' in result and result['errors']:
                    all_errors.extend(result['errors'])
                    # 更新错误类型计数
                    if 'error_summary' in result:
                        for error_type, count in result['error_summary']['error_types'].items():
                            total_error_counts[error_type] = total_error_counts.get(error_type, 0) + count

                if verbose:
                    print(f"  Parse success rate: {result['parse_success_rate']:.2%}")
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")
                    if dynamic_query and result['n_queries_saved'] > 0:
                        print(f"  Queries saved: {result['n_queries_saved']}")
                    if result['cost'] > 0:
                        print(f"  Cost: ${result['cost']:.6f}")

                # 保存检查点
                checkpoint_data = {
                    'run_id': run_id,
                    'llm_name': llm.get_name(),
                    'n_samples': n_samples,
                    'n_queries_per_sample': n_queries_per_sample,
                    'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                    'dynamic_query': dynamic_query,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results,
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'total_tokens': total_tokens,
                    'total_cost': total_cost,
                    'all_errors': all_errors,
                    'total_error_counts': total_error_counts,
                    'query_savings': query_savings
                }

                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2, default=_convert_numpy)

            except Exception as e:
                print(f"  处理样本 {idx + 1} 时出错: {str(e)}")
                traceback.print_exc()
                continue

        # 计算统计量
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

        # 计算p值（单样本t检验，零假设为0）
        def calculate_p_value(rates):
            if not rates or len(rates) < 2:
                return None
            t_stat, p_val = stats.ttest_1samp(rates, 0)
            return p_val

        # 编译最终结果
        final_results = {
            'run_id': run_id,
            'llm_name': llm.get_name(),
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'dynamic_query': dynamic_query,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'max_edges_constraint': self.max_edges,
            'statistics': {
                'parse_success_rate': {
                    **calculate_stats(parse_success_rates),
                    'p_value': calculate_p_value(parse_success_rates)
                },
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
                },
                'query_savings': {  # 新增：查询节省统计
                    **calculate_stats(query_savings),
                    'total': sum(query_savings)
                }
            },
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens,
                'avg_tokens_per_sample': total_tokens / len(all_results) if all_results else 0,
                'avg_tokens_per_query': total_tokens / sum(r['n_queries'] for r in all_results) if all_results else 0
            },
            'cost': {
                'total_cost': total_cost,
                'avg_cost_per_sample': total_cost / len(all_results) if all_results else 0,
                'avg_cost_per_query': total_cost / sum(r['n_queries'] for r in all_results) if all_results else 0,
                'cost_saved_by_dynamic': sum(
                    r['n_queries_saved'] * (total_cost / sum(r['n_queries'] for r in all_results))
                    for r in all_results) if all_results and sum(r['n_queries'] for r in all_results) > 0 else 0
            },
            'error_summary': {
                'total_errors': len(all_errors),
                'error_types': total_error_counts,
                'error_rate': len(all_errors) / sum(r['n_queries'] for r in all_results) if all_results else 0
            },
            'per_sample_results': all_results
        }

        # 打印综合摘要
        print("\n" + "=" * 60)
        print("Enhanced Benchmark Summary")
        print("=" * 60)
        print(f"Evaluated samples: {len(all_results)}/{n_samples}")
        print(f"Max edge constraint: {self.max_edges if self.max_edges is not None else 'unbounded'}")
        print("Dynamic query strategy: Enabled" if dynamic_query else "Dynamic query strategy: Disabled")
        if dynamic_query:
            print(f"Total queries saved by dynamic strategy: {sum(query_savings)}")

        for metric_name, metric_key in [('Parse success rate', 'parse_success_rate'),
                                        ('Valid rate', 'valid_rate'),
                                        ('Novelty rate', 'novelty_rate'),
                                        ('Recovery rate', 'recovery_rate')]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance: {stats_dict['var']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value: {stats_dict['p_value']:.4f}")

        print(f"\nToken usage:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Prompt tokens: {total_prompt_tokens:,}")
        print(f"  Completion tokens: {total_completion_tokens:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")

        print(f"\nCost:")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")
        if dynamic_query and final_results['cost']['cost_saved_by_dynamic'] > 0:
            print(f"  Cost saved by dynamic strategy: ${final_results['cost']['cost_saved_by_dynamic']:.4f}")

        if all_errors:
            print(f"\nErrors:")
            print(f"  Total errors: {len(all_errors)}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if total_error_counts:
                print(f"  Error types:")
                for error_type, count in sorted(total_error_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {error_type}: {count}")

        print("=" * 60)

        # 成功完成后清理检查点文件
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print(f"\nCleanup checkpoint: {checkpoint_file}")
            except Exception:
                pass

        return final_results


def setup_llm(llm_type: str, **kwargs) -> LLMInterface:
    # 保持原LLM设置逻辑不变
    if llm_type == "openai":
        api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required")

        return OpenAILLM(
            model=kwargs.get('model', 'gpt-4o'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    elif llm_type == "anthropic":
        api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key is required")

        return AnthropicLLM(
            model=kwargs.get('model', 'claude-3-opus-20240229'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    elif llm_type == "openrouter":
        api_key = kwargs.get('api_key') or os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key is required")

        return OpenRouterLLM(
            model=kwargs.get('model', 'anthropic/claude-3.5-sonnet'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def parse_n_observations_filter(filter_str: str) -> List[int]:
    # 保持原解析逻辑不变
    if not filter_str:
        return []

    result = []
    parts = filter_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            # 范围格式如 "2-5"
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            result.extend(range(start, end + 1))
        else:
            # 单个值
            result.append(int(part))

    # 去重并排序
    return sorted(list(set(result)))


def parse_gt_filter(filter_str: str) -> Tuple[Optional[int], Optional[int]]:
    # 保持原解析逻辑不变
    if not filter_str:
        return None, None

    # 检查是否为范围（单破折号且不在开头）
    if '-' in filter_str and not filter_str.startswith('-'):
        parts = filter_str.split('-')
        if len(parts) == 2:
            try:
                min_gt = int(parts[0].strip())
                max_gt = int(parts[1].strip())
                return min_gt, max_gt
            except ValueError:
                pass

    # 否则，视为逗号分隔的列表
    try:
        values = []
        for part in filter_str.split(','):
            values.append(int(part.strip()))
        return sorted(values), None
    except ValueError:
        print(f"Warning: Invalid GT filter format: {filter_str}")
        return None, None


def load_config(config_path: str = "config.yaml") -> Dict:
    # 保持原配置加载逻辑不变
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run the enhanced causal discovery benchmark with comprehensive tracking.\n\n"
                    "Features:\n"
                    "- Token usage and cost tracking\n"
                    "- Checkpoint mechanism for resumable runs\n"
                    "- Enhanced error handling\n"
                    "- Statistical analysis of results\n"
                    "- Adaptive queries based on the number of ground truth graphs\n"
                    "- Dynamic query strategy (stop early after all GTs are discovered)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--dataset", required=True, help="Path to the complete causal dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of observation sets to sample")
    parser.add_argument("--n-observations-filter", type=str, default=None,
                        help="Filter dataset by number of observations (e.g. '2,3,5' or '2-5')")
    parser.add_argument("--gt-filter", type=str, default=None, help="Filter dataset by number of ground truth graphs (e.g. '10-16' or '1,2,4')")
    parser.add_argument("--n-queries", type=int, default=None, help="Fixed number of queries per observation set")
    parser.add_argument("--query-multiplier", type=float, default=2.0, help="Multiplier for adaptive query count")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries per query")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    parser.add_argument("--no-dynamic-query", action="store_true", help="Disable dynamic query strategy")  # New argument

    args = parser.parse_args()

    # 处理verbose/quiet标志
    if args.quiet:
        args.verbose = False

    # 加载配置
    config = load_config(args.config)
    llm_type = config.get('llm', {}).get('type', 'openrouter')

    model = config.get('llm', {}).get('models', {}).get(llm_type)
    if not model:
        default_models = {
            'openrouter': 'openai/gpt-3.5-turbo',
            'openai': 'gpt-4o',
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
    checkpoint_dir = args.checkpoint_dir or config.get('benchmark', {}).get('checkpoint_dir', 'checkpoints')
    verbose = args.verbose and config.get('benchmark', {}).get('verbose', True)

    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    # Generate output filename if not specified
    if args.output is None:
        dataset_name = Path(args.dataset).stem
        model_name = Path(model).stem if model else llm_type
        output_pattern = config.get('benchmark', {}).get("output_pattern", "results/{dataset_name}_{model}.json")
        output = output_pattern.format(dataset_name=dataset_name, model=model_name)
    else:
        output = args.output

    # Parse filters if provided
    n_observations_filter = None
    if args.n_observations_filter:
        n_observations_filter = parse_n_observations_filter(args.n_observations_filter)
        print(f"Filtering by number of observations: {n_observations_filter}")

    gt_filter = None
    if args.gt_filter:
        gt_filter = parse_gt_filter(args.gt_filter)
        if gt_filter[0] is not None:
            if gt_filter[1] is not None:
                print(f"Filtering by number of ground truths: [{gt_filter[0]}, {gt_filter[1]}]")
            else:
                print(f"Filtering by number of ground truths: {gt_filter[0]}")

    # Initialize benchmark with filters
    benchmark = CausalBenchmarkEnhanced(args.dataset,
                                        n_observations_filter=n_observations_filter,
                                        gt_filter=gt_filter)

    # Print configuration
    print("\n" + "=" * 60)
    print("Enhanced Causal Benchmark Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"LLM type: {llm_type}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Number of samples: {args.n_samples}")

    if args.n_queries is not None:
        print(f"Number of queries per sample: {args.n_queries} (fixed)")
    else:
        print(f"Number of queries per sample: {args.query_multiplier}x number of ground truths (adaptive)")

    print(f"Dynamic query strategy: {'Disabled' if args.no_dynamic_query else 'Enabled'}")
    print(f"Max retries: {args.max_retries}")
    print(f"Seed: {args.seed}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output file: {output}")
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
        max_retries=args.max_retries,
        dynamic_query=not args.no_dynamic_query  # Pass dynamic query strategy flag
    )

    # Save final results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w') as f:
        json.dump(results, f, indent=2, default=_convert_numpy)

    print(f"\nFinal results saved to: {output}")


if __name__ == "__main__":
    main()

bibtex_entry = """
@article{chen2025hypospace,
  title={HypoSpace:Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination},
  author={Chen, Tingting and Lin, Beibei and Yuan, Zifeng and Zou, Qiran and He, Hongyu and Zhang, Wei-Nan},
  journal={arXiv preprint arXiv:2510.15614},
  year={2025}
}
"""
