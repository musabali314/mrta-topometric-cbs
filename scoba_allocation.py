"""
scoba_allocation.py
-------------------

Stochastic SCoBA-style task allocation for multi-robot systems.

This version is adapted to integrate with:
    - Task class in agents_and_tasks.py
    - Robot objects
    - TopometricMap (region-based travel times)
    - Main PM-CBS simulation pipeline

Key idea:
    For each agent, we build a decision tree of ATTEMPT/SKIP for tasks
    (sorted by deadline). Expected utility is computed using our heuristic
    success probability model.

    If two agents choose the same task → conflict → CBS-style branching:
        Node 1: forbid task for agent_i
        Node 2: forbid task for agent_j

Outputs:
    plans_dict[agent_id] = list of:
         (task_name, start_time, finish_time, expected_contrib)
    total_expected_utility   (float)
"""

import heapq
import math


# ===============================================================
# UTILITY
# ===============================================================

def success_probability(task, travel_time, start_time, finish_time,
                        alpha=0.7, p_min=0.05):
    """
    Better SCoBA success probability model.
    """
    window = max(1e-6, task.deadline - task.release_time)
    lateness = (finish_time - task.release_time) / window
    lateness = max(0.0, min(1.0, lateness))

    # arrival penalty
    base = 1.0 - alpha * lateness

    # travel uncertainty penalty
    travel_penalty = 0.015 * float(travel_time)

    p = base - travel_penalty
    return max(p_min, min(1.0, p))



# ===============================================================
# LOW-LEVEL SCoBA POLICY
# ===============================================================

def plan_for_agent_stochastic(agent_id,
                              agent_start_loc,
                              tasks,
                              excluded_tasks,
                              travel_time_func,
                              alpha=0.7,
                              p_min=0.1):
    """
    Build the ATTEMPT/SKIP decision tree for one agent.

    Parameters:
        agent_id : string
        agent_start_loc : (x,y)
        tasks : list[Task]
        excluded_tasks : set(str)
        travel_time_func : function(locA, locB) → travel_time (float)
        alpha, p_min : SCoBA hyperparameters

    Returns:
        best_plan : list of (task_name, start_t, finish_t, contrib)
        best_value : float
    """

    # Filter and sort tasks
    candidate_tasks = [t for t in tasks if t.name not in excluded_tasks]
    candidate_tasks.sort(key=lambda t: t.deadline)

    best_plan = []
    best_value = 0.0

    def dfs(idx, current_loc, current_time, current_plan, current_val):
        nonlocal best_plan, best_value

        # No more tasks to consider
        if idx == len(candidate_tasks):
            if current_val > best_value:
                best_value = current_val
                best_plan = list(current_plan)
            return

        task = candidate_tasks[idx]

        # OPTION 1 — SKIP
        dfs(idx + 1,
            current_loc,
            current_time,
            current_plan,
            current_val)

        # OPTION 2 — ATTEMPT if feasible
        travel_time = travel_time_func(current_loc, task.location)
        start_time = max(current_time + travel_time, task.release_time)
        finish_time = start_time + task.duration

        # Check deadline feasibility
        if finish_time > task.deadline:
            return

        # Expected utility
        p_s = success_probability(task, travel_time, start_time,
                                  finish_time, alpha, p_min)
        expected_contrib = p_s * task.utility

        dfs(idx + 1,
            task.location,
            finish_time,
            current_plan + [(task.name, start_time, finish_time, expected_contrib)],
            current_val + expected_contrib)

    # Run tree search
    dfs(
        idx=0,
        current_loc=agent_start_loc,
        current_time=0.0,
        current_plan=[],
        current_val=0.0,
    )

    return best_plan, best_value


# ===============================================================
# HIGH-LEVEL SCoBA SOLVER (CBS-style branching)
# ===============================================================

class SCoBASolver:
    def __init__(self, agents, tasks, travel_time_func,
                 alpha=0.7, p_min=0.1):
        """
        Parameters:
            agents : list[Robot] or list[Agent-like]
                Must support agent.id and agent.start_location
            tasks : list[Task]
            travel_time_func : function(locA, locB) → travel_time
                (can use topometric regions)
        """
        self.agents = agents
        self.tasks = tasks
        self.travel_time_func = travel_time_func
        self.alpha = alpha
        self.p_min = p_min

    # -----------------------------------------------------------
    def find_conflict(self, plans):
        """
        Detect if two agents have selected the same task.
        """
        seen = {}
        for aid, plan in plans.items():
            for (task_name, _, _, _) in plan:
                if task_name in seen:
                    return task_name, seen[task_name], aid
                seen[task_name] = aid
        return None

    # -----------------------------------------------------------
    def solve(self):
        """
        Returns:
            (plans_dict, total_expected_value)
        """
        # Constraints: dict(agent_id → set(task_names_forbidden))
        root_constraints = {a.id: set() for a in self.agents}
        root_plans = {}
        root_value = 0.0

        # Compute initial plans
        for agent in self.agents:
            plan, val = plan_for_agent_stochastic(
                agent_id=agent.id,
                agent_start_loc=agent.start_location,
                tasks=self.tasks,
                excluded_tasks=root_constraints[agent.id],
                travel_time_func=self.travel_time_func,
                alpha=self.alpha,
                p_min=self.p_min
            )
            root_plans[agent.id] = plan
            root_value += val

        # Priority queue: (-utility, node_id, constraints, plans)
        heap = []
        node_id = 0
        heapq.heappush(heap, (-root_value, node_id,
                              root_constraints, root_plans))
        node_id += 1

        while heap:
            _, _, constraints, plans = heapq.heappop(heap)

            conflict = self.find_conflict(plans)
            if conflict is None:
                # FOUND best valid solution
                total_val = sum(sum(p_i[3] for p_i in plans[a.id])
                                for a in self.agents)
                return plans, total_val

            task_name, a_i, a_j = conflict

            # ------------------------------------------
            # BRANCH 1 — forbid task for agent_i
            # ------------------------------------------
            new_constraints_1 = {aid: set(cset)
                                 for aid, cset in constraints.items()}
            new_constraints_1[a_i].add(task_name)

            plans_1 = {}
            total_1 = 0.0
            for agent in self.agents:
                plan, val = plan_for_agent_stochastic(
                    agent_id=agent.id,
                    agent_start_loc=agent.start_location,
                    tasks=self.tasks,
                    excluded_tasks=new_constraints_1[agent.id],
                    travel_time_func=self.travel_time_func,
                    alpha=self.alpha,
                    p_min=self.p_min
                )
                plans_1[agent.id] = plan
                total_1 += val

            heapq.heappush(heap, (-total_1, node_id,
                                  new_constraints_1, plans_1))
            node_id += 1

            # ------------------------------------------
            # BRANCH 2 — forbid task for agent_j
            # ------------------------------------------
            new_constraints_2 = {aid: set(cset)
                                 for aid, cset in constraints.items()}
            new_constraints_2[a_j].add(task_name)

            plans_2 = {}
            total_2 = 0.0
            for agent in self.agents:
                plan, val = plan_for_agent_stochastic(
                    agent_id=agent.id,
                    agent_start_loc=agent.start_location,
                    tasks=self.tasks,
                    excluded_tasks=new_constraints_2[agent.id],
                    travel_time_func=self.travel_time_func,
                    alpha=self.alpha,
                    p_min=self.p_min
                )
                plans_2[agent.id] = plan
                total_2 += val

            heapq.heappush(heap, (-total_2, node_id,
                                  new_constraints_2, plans_2))
            node_id += 1

        return None, 0.0  # Should not reach here
