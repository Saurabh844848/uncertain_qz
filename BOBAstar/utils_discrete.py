import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import networkx as nx
import heapq
from sklearn.cluster import KMeans
import itertools  # To cycle through style combinations

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
    def __init__(self, map_qz, start, goal, q_min, q_max, q_act, alpha, beta, discretization_angle, max_risk_limit=np.inf, acceptable_risk_limit=0.0):
        self.map_qz = map_qz
        self.start = start
        self.goal = goal
        self.q_min = q_min
        self.q_max = q_max
        self.q_act = q_act
        self.alpha = alpha
        self.beta = beta
        self.discretization_angle = discretization_angle
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
        num_points_per_circle = len(angles)
        total_nodes = len(self.map_qz) * num_points_per_circle + 2
        node_positions = np.zeros((total_nodes, 4))

        # Use custom keys for start and goal: "s" for start and "g" for goal.
        index_map = {
            (self.start[0], self.start[1], 0, self.q_act): "s",
            (self.goal[0], self.goal[1], len(self.map_qz) + 1, self.q_act): "g"
        }
        reverse_index_map = {
            "s": (self.start[0], self.start[1], 0, self.q_act),
            "g": (self.goal[0], self.goal[1], len(self.map_qz) + 1, self.q_act)
        }

        # Assign start and goal positions in the node array.
        node_positions[0] = [self.start[0], self.start[1], 0, self.q_act]
        node_positions[1] = [self.goal[0], self.goal[1], len(self.map_qz) + 1, self.q_act]

        node_index = 2  # start index for circle nodes
        for circle_index, (cx, cy, radius, radius_in, risk_limit, toggle) in enumerate(self.map_qz, start=1):
            x_vals = cx + radius * np.cos(angles)
            y_vals = cy + radius * np.sin(angles)
            node_positions[node_index:node_index + num_points_per_circle, 0] = x_vals
            node_positions[node_index:node_index + num_points_per_circle, 1] = y_vals
            node_positions[node_index:node_index + num_points_per_circle, 2] = circle_index
            node_positions[node_index:node_index + num_points_per_circle, 3] = self.q_act
            
            for j in range(num_points_per_circle):
                key = (x_vals[j], y_vals[j], circle_index, self.q_act)
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
        soc_i, soc_j = int(node_i[3]), int(node_j[3])
        
        feasible, lamb, line_segment = self._compute_edge_parameters(soc_i, soc_j, distance)
        
        # Check if the SOC requirements immediately rule out feasibility
        if soc_j > min(self.q_max, soc_i + self.beta * distance):
            return feasible, None, line_segment
        
        if feasible:
            cost = lamb * distance
            visibility_graph.add_edge(
                i, j,
                node_i_info=node_i,
                node_j_info=node_j,
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
        soc_i, soc_j = int(node_i[3]), int(node_j[3])
        
        # Retrieve circle parameters corresponding to node_i (assumes node_i[2] is valid)
        circle = self.map_qz[int(node_i[2] - 1)]
        _, _, radius, radius_in, risk_limit, toggle = circle
        pen_dist = abs(sp.sqrt(radius**2 - (distance / 2)**2))
        # print(f"pen_dist:{pen_dist}")
        if pen_dist < radius_in and toggle==0:
            feasible = False
        elif pen_dist < radius_in and toggle==1:
            feasible, lamb, line_segment = self._compute_edge_parameters(soc_i, soc_j, distance)
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
            feasible, lamb, line_segment = self._compute_edge_parameters(soc_i, soc_j, distance)
            
        if feasible:
            cost = lamb * distance
            risk_cost = self._compute_risk_cost(line_segment, pen_dist, distance)
            visibility_graph.add_edge(
                i, j,
                node_i_info=node_i,
                node_j_info=node_j,
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
        for i, node_i in rev_index_map.items():
            for j, node_j in rev_index_map.items():
                if i == j:
                    continue

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
            rev_index_map: Mapping from node index to node tuple.
        """
        # Define the goal node tuple. The circle index for the goal is set to len(qz_circles)+1.
        goal_node = (self.goal[0], self.goal[1], len(self.map_qz) + 1, self.q_act)
        start_node= (self.start[0], self.start[1], 0, self.q_act)
        
        for i, node in rev_index_map.items():
            # Compute Euclidean distance using sympy (for consistency with the rest of the code)
            distance_goal = sp.sqrt((goal_node[0] - node[0])**2 + (goal_node[1] - node[1])**2)
            distance_start = sp.sqrt((start_node[0] - node[0])**2 + (start_node[1] - node[1])**2)
            heuristic_cost_forward = (self.alpha / (self.alpha + self.beta)) * distance_goal
            heuristic_cost_backward = (self.alpha / (self.alpha + self.beta)) * distance_start
            
            self.visibility_graph.nodes[i]['heuristic_cost'] = { "forward": {"fuel_cost": heuristic_cost_forward, "risk_cost": 0}, "backward": {"fuel_cost": heuristic_cost_backward, "risk_cost": 0}  }
            self.visibility_graph.nodes[i]['upper_bound_cost'] = { "forward": {"fuel_cost": np.inf, "risk_cost": np.inf}, "backward": {"fuel_cost": np.inf, "risk_cost": np.inf}  }

class BidictionalSearch():
    """
    Contains all the functions for the bidirectional biobjective A* search.
    """
    def __init__(self, ):
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
    fuel_costs = [sol[0][0] for sol in solutions]
    risk_costs = [sol[0][1] for sol in solutions]
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
    for current_node, parent_node in solutions[goal_state]:
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

from matplotlib.patches import Circle

def plot_map_with_path(graph_object, Map_qz, start, goal, all_path, reverse_index_map):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    
    ### find the min and max x and y values, for axis limit 
    min_x = min( [min(Map_qz, key=lambda x: x[0]-x[2])[0] - min(Map_qz, key=lambda x: x[0]-x[2])[2], start[0], goal[0]] )
    max_x = max( [max(Map_qz, key=lambda x: x[0]+x[2])[0] + max(Map_qz, key=lambda x: x[0]+x[2])[2], start[0], goal[0]] )
    min_y = min( [min(Map_qz, key=lambda x: x[1]-x[2])[1] - min(Map_qz, key=lambda x: x[1]-x[2])[2], start[1], goal[1]] )
    max_y = max( [max(Map_qz, key=lambda x: x[1]+x[2])[1] + max(Map_qz, key=lambda x: x[1]+x[2])[2], start[1], goal[1]] )
    
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '*', '^', 'D', 'v', 'P', 'X', '<', '>', 'H']
    
    style_combinations = list(itertools.product(markers, line_styles))
    
    path_coordinates = []
    path_coordinates_gas = []
    path_coordinates_electric = []
    
    for index, path in enumerate(all_path):
        marker, line_style = style_combinations[index]

        for path_index in range(len(path) - 2): # for each node in the path:
            i = path[path_index]
            j = path[path_index + 1]
            info_graph = graph_object.visibility_graph[i][j]
            node_i_coor = np.array(info_graph['node_i_info'][:2])
            node_j_coor = np.array(info_graph['node_j_info'][:2])
            for line_segment_type, line_segment_value in info_graph['line_segment'].items():
                if line_segment_type[0] == 'g':
                    temp_coor_1 = list(node_i_coor + line_segment_value[0] * (node_j_coor - node_i_coor))
                    temp_coor_2 = list(node_i_coor + line_segment_value[1] * (node_j_coor - node_i_coor))
                    result = [list(pair) for pair in zip(temp_coor_1, temp_coor_2)]
                    path_coordinates_gas.append(result)
                elif line_segment_type[0] == 'e':
                    temp_coor_1 = list(node_i_coor + line_segment_value[0] * (node_j_coor - node_i_coor))
                    temp_coor_2 = list(node_i_coor + line_segment_value[1] * (node_j_coor - node_i_coor))
                    result = [list(pair) for pair in zip(temp_coor_1, temp_coor_2)]
                    path_coordinates_electric.append(result)
        
        ### Draw the path
        for path_coor in path_coordinates_gas:
            ax.plot(path_coor[0], path_coor[1], 'r', linestyle=line_style, marker=marker, markersize=3)
        for path_coor in path_coordinates_electric:
            ax.plot(path_coor[0], path_coor[1], 'g', linestyle=line_style, marker=marker, markersize=3)
    
    for circle_info in Map_qz:
        # Create a circle patch
        circle_outer = Circle((circle_info[0], circle_info[1]), radius=circle_info[2], fill=True, facecolor=(0, 0, 1, 0.2), edgecolor='blue', linewidth=2, zorder=1)
        circle_inner = Circle((circle_info[0], circle_info[1]), radius=circle_info[3], fill=True, facecolor=(0, 0, 0, 0.5), edgecolor='black', linewidth=2, zorder=2)

        # Add the circle to the Axes
        ax.add_patch(circle_inner)
        ax.add_patch(circle_outer)

    # Set equal aspect so circles look circular
    ax.set_aspect('equal', 'box')        
    
    # Set axis limits
    ax.set_xlim(min_x - (max_x - min_x)/10, max_x + (max_x - min_x)/10)
    ax.set_ylim(min_y - (max_y - min_y)/10, max_y + (max_y - min_y)/10)

    # Plot the start and goal points
    ax.plot(start[0], start[1], 'ro', label='Start')
    ax.plot(goal[0], goal[1], 'go', label='Goal')

    # Add a legend
    ax.legend()
    plt.title("Pareto optimal paths")
    plt.grid()
    plt.show()
    
def sample_representative_paths(path_sorted, num_path):
    # If we only need 2 points, return min and max.
    if num_path <= 2:
        return [path_sorted[0], path_sorted[-1]]
    
    fuel_costs_arr = np.array([path[-1][0] for path in path_sorted])

    # Reshape the data for k-means clustering (n_samples, n_features)
    X = fuel_costs_arr.reshape(-1, 1)
    
    # Run k-means with the desired number of clusters
    kmeans = KMeans(n_clusters=num_path, random_state=0)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_
    
    # For each cluster, select the data point closest to its center.
    representatives = []
    for cluster in range(num_path):
        # Find indices of data points in the current cluster
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) == 0:
            continue  # This cluster got no points, skip it.
        cluster_points = fuel_costs_arr[cluster_indices]
        # Compute the absolute difference from the cluster center
        diff = np.abs(cluster_points - centers[cluster])
        # Select the point closest to the center
        rep  = cluster_points[np.argmin(diff)]
        representatives.append(rep)
    
    # Sort the representative points to maintain order
    representatives = np.array(representatives)
    representatives.sort()
    
    # Force inclusion of extreme values (min and max)
    representatives[0] = path_sorted[0][-1][0]
    representatives[-1] = path_sorted[-1][-1][0]
    
    ### Get the sampled representative path
    representative_paths =  [ path_sorted[int(np.where(fuel_costs_arr == fuel_cost_rep)[0][0])] for fuel_cost_rep in representatives ]
    return representative_paths