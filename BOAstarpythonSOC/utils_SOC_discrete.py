import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import networkx as nx
import heapq

class GraphConstructionDiscretization:
    """
    Constructs a graph based on circle discretization and edge feasibility.
    
    Attributes:
        map_qz (list of tuples): Each tuple (cx, cy, radius) represents a circle.
        start (tuple): (x, y) coordinates of the start point.
        goal (tuple): (x, y) coordinates of the goal point.
        q_min, q_max, q_act: Numeric parameters.
        alpha, beta: Parameters for feasibility computations.
        discretization_angle (float): Angle step (in degrees) for discretization.
    """
    def __init__(self, map_qz, start, goal, q_min, q_max, q_act, alpha, beta, discretization_angle, discretization_SOC, max_risk_limit, acceptable_risk_limit):
        self.map_qz = map_qz
        self.start = start
        self.goal = goal
        self.q_min = q_min
        self.q_max = q_max
        self.q_act = q_act
        self.alpha = alpha
        self.beta = beta
        self.discretization_angle = discretization_angle
        self.discretization_SOC = discretization_SOC
        self.max_risk_limit = max_risk_limit
        self.acceptable_risk_limit = acceptable_risk_limit

    def create_nodes(self):
        """
        Discretizes the circles and creates nodes with start labeled as "s" and goal as "g".
        
        Returns:
            node_positions (np.ndarray): Each row is [x, y, circle_index, q_act].
            index_map (dict): Maps (x, y, circle_index, q_act) to a node key.
            reverse_index_map (dict): Reverse mapping from node key to (x, y, circle_index, q_act).
        """
        angles = np.deg2rad(np.arange(0, 360 + self.discretization_angle, self.discretization_angle))
        self.SOC = np.arange(self.q_min, self.q_max + self.discretization_SOC, self.discretization_SOC)
        self.num_SOC = len(self.SOC)
        num_points_per_circle = len(angles)*self.num_SOC
        total_nodes = len(self.map_qz) * num_points_per_circle + 2*self.num_SOC
        node_positions = np.zeros((total_nodes, 4))

        # Use custom keys for start and goal: "s" for start and "g" for goal.
        index_map = {
            (self.start[0], self.start[1], 0, SOC_charge): i for i, SOC_charge in enumerate(self.SOC)
                    }
        
        index_map.update({
            (self.goal[0], self.goal[1], len(self.map_qz) + 1, SOC_charge): i+self.num_SOC for i, SOC_charge in enumerate(self.SOC)
        })
        
        reverse_index_map = {value: key for key, value in index_map.items()}

        # Assign start and goal positions in the node array.
        node_positions[:self.num_SOC, 0] = self.start[0]
        node_positions[:self.num_SOC, 1] = self.start[1]
        node_positions[:self.num_SOC, 2] = 0
        node_positions[:self.num_SOC, 3] = self.SOC
        
        node_positions[self.num_SOC:self.num_SOC*2, 0] = self.goal[0]
        node_positions[self.num_SOC:self.num_SOC*2, 1] = self.goal[1]
        node_positions[self.num_SOC:self.num_SOC*2, 2] = len(self.map_qz) + 1
        node_positions[self.num_SOC:self.num_SOC*2, 3] = self.SOC
        node_index = 2*self.num_SOC  # start index for circle nodes
        
        for circle_index, (cx, cy, radius, radius_in, risk_limit, toggle) in enumerate(self.map_qz, start=1):
            x_vals = cx + radius * np.cos(angles)
            y_vals = cy + radius * np.sin(angles)
            x_vals = np.repeat(x_vals, self.num_SOC)
            y_vals = np.repeat(y_vals, self.num_SOC)
            SOC_reps = np.tile(self.SOC, len(angles))
            
            node_positions[node_index:node_index + num_points_per_circle, 0] = x_vals
            node_positions[node_index:node_index + num_points_per_circle, 1] = y_vals
            node_positions[node_index:node_index + num_points_per_circle, 2] = circle_index
            node_positions[node_index:node_index + num_points_per_circle, 3] = SOC_reps
            
            for j in range(num_points_per_circle):
                key = (x_vals[j], y_vals[j], circle_index, SOC_reps[j])
                index_map[key] = node_index + j
                reverse_index_map[node_index + j] = key
            node_index += num_points_per_circle

        self.node_positions = node_positions
        return node_positions, index_map, reverse_index_map

    @staticmethod
    def point_to_segment_distance(px, py, x1, y1, x2, y2):
        """
        Computes the minimum distance from point (px, py) to the line segment defined by (x1, y1) and (x2, y2).
        """
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D
        param = dot / len_sq if len_sq != 0 else -1

        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx, yy = x1 + param * C, y1 + param * D

        dx = px - xx
        dy = py - yy
        return np.sqrt(dx * dx + dy * dy)
    
    def line_segment_intersects_circle(self, circle, point1, point2):
        """
        Checks if the line segment (point1 to point2) intersects the given circle.
        
        Args:
            circle (tuple): (center_x, center_y, radius)
            point1 (tuple): (x1, y1)
            point2 (tuple): (x2, y2)
            
        Returns:
            bool: True if the segment intersects the circle, False otherwise.
        """
        center_x, center_y, radius, radius_in, risk_limit, toggle = circle
        x1, y1 = point1
        x2, y2 = point2
        return self.point_to_segment_distance(center_x, center_y, x1, y1, x2, y2) < radius

    def _compute_edge_parameters(self, soc_i, soc_j, distance):
        """
        Computes feasibility, lambda (λ) value, and line segment parameters for an edge.
        
        Args:
            soc_i (float): State of charge at node i.
            soc_j (float): State of charge at node j.
            distance (float): Euclidean distance between node i and node j.
            
        Returns:
            tuple: (feasible (bool), lamb (float or None), line_segment (dict))
        """
        feasible = False
        line_segment = {}
        
        # Check if the SOC requirements immediately rule out feasibility
        if soc_j > min(self.q_max, soc_i + self.beta * distance):
            return feasible, None, line_segment
        
        # If the drop is sufficiently steep that the whole edge is "feasible"
        if (soc_i - self.alpha * distance) < self.q_min and soc_j <= soc_i - self.alpha * distance:
            feasible = True
            lamb = 1
            line_segment = {"e": (0, lamb)}
        else:
            lamb_1 = (self.q_max - soc_i) / (self.beta * distance)
            lamb_2 = (self.q_max - soc_j) / (self.alpha * distance)
            
            if lamb_1 >= 0 and lamb_2 >= 0 and (lamb_1 + lamb_2 <= 1):
                lamb = lamb_1 + self.alpha / (self.beta + self.alpha) * (1 - lamb_1 - lamb_2)
                lamb_2_sw = lamb_2 + self.beta / (self.beta + self.alpha) * (1 - lamb_1 - lamb_2)
                # Note: using the key "g" twice in a dict will override; if multiple segments are needed,
                # consider using a list of segments.
                line_segment = {"g": (0, lamb_1), "e": (lamb_1, lamb_2_sw), "g2": (lamb_2_sw, 1)}
            else:
                lamb = (soc_j - soc_i + self.alpha * distance) / ((self.alpha + self.beta) * distance)
                line_segment = {"g": (0, lamb), "e": (lamb, 1)}
            if 0 <= lamb <= 1:
                feasible = True
                
        return feasible, lamb, line_segment

    def external_edge_addition(self, i, node_i, j, node_j, visibility_graph):
        """
        Adds an external edge to the visibility graph if the edge is feasible.
        
        Args:
            i (int): Node index for node_i.
            node_i (array-like): Information for node i.
            j (int): Node index for node_j.
            node_j (array-like): Information for node j.
            visibility_graph (networkx.Graph): The graph to update.
            
        Returns:
            networkx.Graph: The updated visibility graph.
        """
        distance = np.linalg.norm(np.array(node_i[:2]) - np.array(node_j[:2]))
        for i_add, SOC_i in enumerate(self.SOC):
            for j_add, SOC_j in enumerate(self.SOC):     
                feasible, lamb, line_segment = self._compute_edge_parameters(SOC_i, SOC_j, distance)
                # print(f"i: {i+i_add}, j: {j+j_add}")
                if feasible:
                    # print(f"i: {i+i_add}, j: {j+j_add}")
                    cost = lamb * distance
                    visibility_graph.add_edge(
                        i+i_add, j+j_add,
                        node_i_info= (node_i[0], node_i[1], node_i[2], SOC_i),
                        node_j_info= (node_j[0], node_j[1], node_j[2], SOC_j),
                        line_segment=line_segment,
                        fuel_cost=cost,
                        risk_cost=0,
                        feasibility=feasible,
                        edge_type="external"
                    )
        return visibility_graph

    def internal_edge_addition(self, i, node_i, j, node_j, visibility_graph):
        """
        Adds an internal edge (within a circle) to the visibility graph if the edge is feasible.
        Also computes a risk cost for the edge.
        
        Args:
            i (int): Node index for node_i.
            node_i (array-like): Information for node i.
            j (int): Node index for node_j.
            node_j (array-like): Information for node j.
            visibility_graph (networkx.Graph): The graph to update.
            
        Returns:
            networkx.Graph: The updated visibility graph.
        """
        distance = np.linalg.norm(np.array(node_i[:2]) - np.array(node_j[:2]))
        # Retrieve circle parameters corresponding to node_i (assumes node_i[2] is valid)
        circle = self.map_qz[int(node_i[2] - 1)]
        _, _, radius, radius_in, risk_limit, toggle = circle
        pen_dist = abs(sp.sqrt(radius**2 - (distance / 2)**2))
        
        for i_add, SOC_i in enumerate(self.SOC):
            for j_add, SOC_j in enumerate(self.SOC):                
                if pen_dist < radius_in and toggle==0:
                    feasible = False
                elif pen_dist < radius_in and toggle==1:
                    feasible, lamb, line_segment = self._compute_edge_parameters(SOC_i, SOC_j, distance)
                    if feasible:
                        r_minus, r_plus = 0.5 - radius_in/(2*radius), 0.5 + radius_in/(2*radius)
                        l_minus, l_plus = line_segment["e"]
                        electric_len_req = 2*sp.sqrt(radius_in**2 - pen_dist**2)
                        
                        if (l_plus-l_minus) >= electric_len_req and feasible:
                            if (l_minus <= r_minus) and (l_plus >= r_plus):
                                feasible = True
                            elif (l_minus >= r_minus) and (l_plus >= r_plus):
                                line_segment["g"] = (0, r_minus)
                                line_segment["e"] = (r_minus, r_minus+(l_plus-l_minus))
                                line_segment["g2"] = (r_minus+(l_plus-l_minus), 1)
                            else:
                                feasible = False
                        else:
                            feasible = False
                else:
                    feasible, lamb, line_segment = self._compute_edge_parameters(SOC_i, SOC_j, distance)
                    
                if feasible:
                    cost = lamb * distance
                    risk_cost = self._compute_risk_cost(line_segment, pen_dist, distance)
                    visibility_graph.add_edge(
                        i+i_add, j+j_add,
                        node_i_info= (node_i[0], node_i[1], node_i[2], SOC_i),
                        node_j_info= (node_j[0], node_j[1], node_j[2], SOC_j),
                        line_segment=line_segment,
                        fuel_cost=cost,
                        risk_cost=risk_cost,
                        feasibility=feasible,
                        edge_type="internal"
                        )
        return visibility_graph

    def _compute_risk_cost(self, line_segment, pen_dist, distance):
        """
        Computes the risk cost for an internal edge based on the line segment parameters.
        
        Args:
            line_segment (dict): Contains segments with keys (e.g. "g", "e").
            pen_dist: Penetration distance (symbolic) computed from the circle.
            distance (float): Euclidean distance between the two nodes.
            
        Returns:
            risk_cost (sympy expression): The computed risk cost.
        """
        risk_cost = 0
        constant_factor = 30
        factor = 1 / (pen_dist + 0.01)
        
        for key, (lamb_start, lamb_end) in line_segment.items():
            if key == "e":
                continue  # 'e' segments do not contribute to risk cost
            elif key.startswith("g"):
                if lamb_start <= 0.5 <= lamb_end:
                    risk_cost += abs(
                        factor * (
                            sp.atan((distance/2 - lamb_start * distance) / (pen_dist + 0.01)) +
                            sp.atan((lamb_end * distance - distance/2) / (pen_dist + 0.01))
                        )
                    )
                elif lamb_start >= 0.5:
                    risk_cost += abs(
                        factor * (
                            sp.atan((lamb_end * distance - distance/2) / (pen_dist + 0.01)) -
                            sp.atan((lamb_start * distance - distance/2) / (pen_dist + 0.01))
                        )
                    )
                elif lamb_end <= 0.5:
                    risk_cost += abs(
                        factor * (
                            sp.atan((distance/2 - lamb_start * distance) / (pen_dist + 0.01)) -
                            sp.atan((distance/2 - lamb_end * distance) / (pen_dist + 0.01))
                        )
                    )
        return constant_factor * risk_cost

    def build_visibility_graph(self, rev_index_map):
        """
        Builds the visibility graph by adding internal and external edges between nodes.
        
        Args:
            graph_obj: Instance of GraphConstructionDiscretization.
            rev_index_map: Mapping of node indices to node information.
            
        Returns:
            visibility_graph: A directed graph (nx.DiGraph) with added edges.
        """
        self.visibility_graph = nx.DiGraph()
        # Iterate over all pairs of nodes
        for i in range(0, len(rev_index_map), self.num_SOC):
            for j in range(0, len(rev_index_map), self.num_SOC):
                if i == j:
                    continue
                node_i, node_j = rev_index_map[i], rev_index_map[j]
                
                circle_i, circle_j = node_i[2], node_j[2]
                x_i, y_i = node_i[0], node_i[1]
                x_j, y_j = node_j[0], node_j[1]

                # Case 1: Internal edge if both nodes are from the same circle
                if circle_i == circle_j:
                    self.internal_edge_addition(i, node_i, j, node_j, self.visibility_graph)
                else:
                    # For external edges, check if any QZ circle intersects the line segment
                    intersection_exists = False
                    for circle in self.map_qz:
                        if self.line_segment_intersects_circle(circle, (x_i, y_i), (x_j, y_j)):
                            intersection_exists = True
                            break
                    if not intersection_exists:
                        self.external_edge_addition(i, node_i, j, node_j, self.visibility_graph)

    def assign_heuristic_costs(self, rev_index_map):
        """
        Assigns a heuristic cost for each node based on its Euclidean distance to the goal.
        
        Args:
            visibility_graph: The graph with nodes.
            rev_index_map: Mapping from node index to node tuple.
            goal: Goal point as (x, y).
            qz_circles: List of QZ circles.
            q_act: The state-of-charge value.
            alpha: UAV parameter.
            beta: UAV parameter.
        """
        # Define the goal node tuple. The circle index for the goal is set to len(qz_circles)+1.
        goal_node = (self.goal[0], self.goal[1], len(self.map_qz) + 1, self.q_max)
        for node in self.visibility_graph.nodes:
            node_info = rev_index_map[node]
            # Compute Euclidean distance using sympy (for consistency with the rest of the code)
            distance = sp.sqrt((goal_node[0] - node_info[0])**2 + (goal_node[1] - node_info[1])**2)
            heuristic_cost = (self.alpha / (self.alpha + self.beta)) * distance
            self.visibility_graph.nodes[node]['heuristic_cost'] = heuristic_cost


# =============================================================================
# Biobjective Optimization (Search)
# =============================================================================
# visibility_graph = graph_object.visibility_graph

def biobjective_search( graph_object, start_state=2, goal_state=[3,4,5], reduce_factor=1):
    """
    Performs a biobjective search (fuel cost and risk cost) over the visibility graph.
    
    Args:
        graph_object: The graph object containing the visibility graph, and other relevant information.
        start_state: Identifier for the start node (default "s").
        goal_state: Identifier for the goal node (default "g").
        reduce_factor: A factor used in pruning dominated paths.
        
    Returns:
        sols: A dict mapping each state to a list of solution tuples.
        g2_min: A dict mapping each state to its minimum risk cost.
    """
    visibility_graph = graph_object.visibility_graph
    max_risk_limit, acceptable_risk_limit = graph_object.max_risk_limit, graph_object.acceptable_risk_limit
    
    all_states = list(visibility_graph.nodes)
    # print("All states:", all_states)
    sols = {state: [] for state in all_states}
    g2_min = {state: np.inf for state in all_states}
    open_set = []

    # The start node is represented as a tuple: (f1, f2, g1, g2, state)
    # and its parent is set to None.
    start_node = (0, 0, 0, 0, start_state)
    heapq.heappush(open_set, [start_node, (None, None, None, None, None)])
    g2_min_goal = np.inf
    while open_set:
        current_node, parent_node = heapq.heappop(open_set)
        current_f1, current_f2, current_g1, current_g2, current_state = current_node
        length_SOC = len(goal_state)
        g2_min_goal = min( [ value for key, value in g2_min.items() if key in goal_state ] )
        # Prune if the current risk cost is dominated
        print(f"Current_state: {current_state}")
        if (current_g2 >= reduce_factor * g2_min[current_state] or
            current_f2 >= reduce_factor * g2_min_goal):
            continue
        
        g2_min[current_state] = current_g2
        sols[current_state].append([current_node, parent_node])

        if current_state in goal_state and current_g2 < acceptable_risk_limit:
            return sols, g2_min
        
        # Stop expanding if the goal is reached
        if current_state in goal_state:
            continue

        # Expand successors of the current state
        for successor in visibility_graph.successors(current_state):
            edge_data = visibility_graph.edges[current_state, successor]
            g1 = current_g1 + edge_data['fuel_cost']
            f1 = g1 + visibility_graph.nodes[successor]['heuristic_cost']
            g2 = current_g2 + edge_data['risk_cost']
            # For risk, we use a zero heuristic.
            f2 = g2

            # Prune dominated successors
            if (g2 >= reduce_factor * g2_min[successor] or
                f2 >= reduce_factor * g2_min_goal) or (current_g2 > max_risk_limit):
                continue
            
            child_node = (f1, f2, g1, g2, successor)
            heapq.heappush(open_set, [child_node, current_node])

    return sols, g2_min

def extract_costs(solutions, target_state):
    """
    Extracts fuel and risk cost values for a given target state from the solutions.
    
    Args:
        solutions: Dict mapping states to solution tuples.
        target_state: The state for which to extract cost values.
        
    Returns:
        fuel_costs: List of fuel cost values.
        risk_costs: List of risk cost values.
    """
    fuel_costs = [sol[0][0] for target in target_state for sol in solutions[target]]
    risk_costs = [sol[0][1] for target in target_state for sol in solutions[target]]
    return fuel_costs, risk_costs


def check_pareto_optimality(fuel_costs, risk_costs):
    """
    Checks for dominance among the solutions and returns the indices of dominated solutions.
    
    Args:
        fuel_costs: List of fuel cost values.
        risk_costs: List of risk cost values.
        
    Returns:
        dominated_indices: List of indices corresponding to dominated solutions.
    """
    dominated_indices = []
    n = len(fuel_costs)
    for i in range(n):
        for j in range(i + 1, n):
            if fuel_costs[i] <= fuel_costs[j] and risk_costs[i] <= risk_costs[j]:
                dominated_indices.append(i)
            elif fuel_costs[i] >= fuel_costs[j] and risk_costs[i] >= risk_costs[j]:
                dominated_indices.append(j)
    return dominated_indices

def plot_costs(fuel_costs, risk_costs):
    """
    Plots a scatter plot of fuel cost versus risk cost.
    
    Args:
        fuel_costs: List of fuel cost values.
        risk_costs: List of risk cost values.
    """
    plt.figure(figsize=(14, 10))
    plt.scatter(fuel_costs, risk_costs, color='royalblue', edgecolors='black',
                alpha=0.75, s=80)
    plt.xlabel("Fuel Cost", fontsize=14, fontweight='bold')
    plt.ylabel("Risk Cost", fontsize=14, fontweight='bold')
    plt.title("Fuel Cost vs. Risk Cost", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    
# =============================================================================
# Reconstructing Solution Paths
# =============================================================================
def reconstruct_solution_paths(solutions, start_state="s", goal_state="g"):
    """
    Reconstructs paths from the start state to the goal state using the parent links stored in solutions.
    
    Each solution is a tuple of the current node and its parent. The node tuple is in the form:
        (f1, f2, g1, g2, state)
    
    Returns:
        solution_paths: A list of paths. Each path is a list of states followed by a cost tuple.
    """
    solution_paths = []
    # Iterate over each solution for the goal state
    for goal_index in goal_state:
        print(f"goal index: {goal_index}")
        for current_node, parent_node in solutions[goal_index]:
            path = [current_node[-1]]  # start with the goal state
            cost = (current_node[0], current_node[1])
            # Traverse backward through the parent links
            parent_state = parent_node[-1] if parent_node[4] is not None else None
            curr_node, par_node = current_node, parent_node
            while parent_state is not None and parent_state != start_state:
                found = False
                for sol in solutions[parent_state]:
                    if sol[0] == par_node:
                        curr_node, par_node = sol
                        path.append(curr_node[-1])
                        parent_state = par_node[-1]
                        found = True
                        break
                if not found:
                    break
            if par_node is not None:
                path.append(par_node[-1])
            path.reverse()
            # Append the associated cost tuple at the end of the path
            path.append(cost)
            solution_paths.append(path)
    return solution_paths