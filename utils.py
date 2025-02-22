import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sympy
from sympy import symbols, Eq, solve
from sympy import simplify

class plot_map:
    
    def __init__(self, Map_qz, start, goal):
        self.Map_qz = Map_qz
        self.start = start
        self.goal = goal
    
    def plot_map_circle(self, tangents=False, tangent_points= [[(0,0), (0,0)]]):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)
        
        ### find the min and max x and y values, for axis limit 
        min_x = min( [min(self.Map_qz, key=lambda x: x[0]-x[2])[0] - min(self.Map_qz, key=lambda x: x[0]-x[2])[2], self.start[0], self.goal[0]] )
        max_x = max( [max(self.Map_qz, key=lambda x: x[0]+x[2])[0] + max(self.Map_qz, key=lambda x: x[0]+x[2])[2], self.start[0], self.goal[0]] )
        min_y = min( [min(self.Map_qz, key=lambda x: x[1]-x[2])[1] - min(self.Map_qz, key=lambda x: x[1]-x[2])[2], self.start[1], self.goal[1]] )
        max_y = max( [max(self.Map_qz, key=lambda x: x[1]+x[2])[1] + max(self.Map_qz, key=lambda x: x[1]+x[2])[2], self.start[1], self.goal[1]] )
        
        
        for circle_info in self.Map_qz:
            # Create a circle patch
            print(circle_info)
            circle = Circle((circle_info[0], circle_info[1]), radius=circle_info[2], fill=False, edgecolor='blue', linewidth=2)

            # Add the circle to the Axes
            ax.add_patch(circle)

        if tangents:
            # Plot the tangent lines
            for tangent_point in tangent_points:
                # print(tangent_point)
                ax.plot( (tangent_point[0][0], tangent_point[1][0]), (tangent_point[0][1], tangent_point[1][1]), 'b--')
                # ax.plot( (tangent_point[2][0], tangent_point[3][0]), (tangent_point[2][1], tangent_point[3][1]), 'b--')
                ax.plot( (tangent_point[0][0], tangent_point[1][0]), (tangent_point[0][1], tangent_point[1][1]), 'bo')
                # ax.plot( (tangent_point[2][0], tangent_point[3][0]), (tangent_point[2][1], tangent_point[3][1]), 'bo')
        # Set equal aspect so circles look circular
        ax.set_aspect('equal', 'box')        
        
        # Set axis limits
        ax.set_xlim(min_x - (max_x - min_x)/10, max_x + (max_x - min_x)/10)
        ax.set_ylim(min_y - (max_y - min_y)/10, max_y + (max_y - min_y)/10)

        # Plot the start and goal points
        ax.plot(self.start[0], self.start[1], 'ro', label='Start')
        ax.plot(self.goal[0], self.goal[1], 'go', label='Goal')

        # Add a legend
        ax.legend()
        plt.title("Circle Using Patches")
        plt.show()


class tangent_line:
    
    def __init__(self, Map_qz, start, goal):
        self.Map_qz = Map_qz
        self.start = start
        self.goal = goal
        

    def tangent_from_pt_to_circle(self, point, circle):
        """
        Returns the equations (in symbolic form) of the tangents
        from (x1, y1) to the circle with center (x0, y0) and radius r.
        """
        x1, y1 = point
        x0, y0, r = circle
        
        # Define symbolic variables
        x, y, m = symbols('x y m', real=True)

        # We'll treat x1, y1, r as known constants and substitute them in later.
        # Equation of line in slope-intercept form: y = m*x + c
        # But c = y1 - m*x1 (so that line passes through (x1, y1))
        c_expr = y1 - m*x1

        # Circle eq: (x - x0)^2 + (y - y0)^2 = r^2
        # Substitute y = m*x + c_expr into (x - x0)^2 + (y - y0)^2 - r^2 = 0
        expr = (x - x0)**2 + (m*x + c_expr - y0)**2 - r**2
        circle_poly = sympy.expand(expr)

        # Extract coefficients of x^2, x^1, and x^0
        A = circle_poly.coeff(x, 2)
        B = circle_poly.coeff(x, 1)
        C = circle_poly.coeff(x, 0)

        # Discriminant
        disc = B**2 - 4*A*C
        
        # Solve discriminant == 0 for m (slopes of tangents)
        disc_eq = Eq(disc, 0)
        m_solutions = solve(disc_eq, m, dict=True)
        
        tangent_lines = []
        for sol in m_solutions:
            slope_expr = sol[m]

            # For a tangent, x_tangent satisfies derivative of the polynomial = 0 or simply the single solution in x.
            # One way (shown here) is: x_tangent = -B/(2A) AFTER substituting the slope back in.
            # Make sure to substitute slope_expr in B and A before dividing:
            A_sub = A.subs(m, slope_expr)
            B_sub = B.subs(m, slope_expr)

            x_tangent_point_on_circle = - B_sub / (2*A_sub)

            # IMPORTANT FIX:
            # We cannot do m(x_tangent_point_on_circle).  m is a symbol, slope_expr is a Sympy expression.
            # So the correct y-value is slope_expr * x_tangent_point_on_circle + c_expr (substituted).
            y_tangent_point_on_circle = slope_expr * x_tangent_point_on_circle + c_expr.subs(m, slope_expr)
            tangent_point_on_circle = (simplify(x_tangent_point_on_circle), simplify(y_tangent_point_on_circle))
            
            # Convert from y = m*x + c to ax + by + c = 0
            # => y - m*x - c = 0  =>  (-m)x + 1*y - c = 0
            # but for clarity we keep a = m, b = -1, c = ...
            line_dict = {
                "line_eq": {
                    "a": simplify(slope_expr), 
                    "b": -1, 
                    "c": simplify(c_expr.subs(m, slope_expr))
                },
                "tangent_point_circle1": point, 
                "tangent_point_circle2": tangent_point_on_circle,
                "circle1": (point[0], point[1], 0),
                "circle2": circle,
                "tangent_type": "point2circle"
            }
            tangent_lines.append(line_dict)
            
        return tangent_lines
    
    def tangent_intersection(self, tangent, tangent_circle=()):
        """
        Checks for circles (other than tangent_circle) that this tangent line intersects.
        If the perpendicular distance from the circle center to the line is < circle radius,
        then the circle is intersected.
        """
        intersecting_circles = []
        for circle in self.Map_qz:
            if circle != tangent_circle:
                dist = abs(tangent["a"]*circle[0]  + tangent["b"]*circle[1] + tangent["c"]) / sympy.sqrt(tangent["a"]**2 + tangent["b"]**2)
                if dist < circle[2]:
                    intersecting_circles.append(circle)
        # print(f"INtersecting circles: {intersecting_circles}")
        return intersecting_circles 
    
    def line_circle_intersection_points(self, line, circle, point):
        """
        Finds the intersection(s) of a line with a given circle, then returns whichever
        intersection is closest to 'point'.
        """
        x1, y1 = point
        x0, y0, r = circle
        
        # Define x symbol
        x = symbols('x', real=True)

        # line["line_eq"] is in form: a*x + b*y + c = 0
        # => y = -a/b * x - c/b, but from your stored form it looks like y = a*x + c if b=-1.
        a_ = line["line_eq"]["a"]
        b_ = line["line_eq"]["b"]
        c_ = line["line_eq"]["c"]
        
        # If your line is strictly y = a*x + c, you have:
        # y = (a_)*x + c_,   where b_ = -1 to match (a_)x + (-1)*y + c_ = 0.
        # Now circle eq: (x - x0)^2 + (y - y0)^2 = r^2
        # Sub y in:
        expr = (x - x0)**2 + (a_*x + c_ - y0)**2 - r**2
        intersecting_points_x = solve(expr, x)
        
        # There could be 0,1, or 2 solutions:
        if not intersecting_points_x:
            return None  # No intersection

        # Build possible intersection points (x_val, y_val)
        possible_points = []
        for x_val in intersecting_points_x:
            y_val = a_*x_val + c_
            possible_points.append((x_val, y_val))
        
        # Return the one closest to (x1, y1)
        intersecting_point = min(
            possible_points,
            key=lambda pt: (pt[0] - x1)**2 + (pt[1] - y1)**2
        )
        
        return intersecting_point
    
    def tangent_no_intersection(self, intersecting_circles, tangent, tangent_circle):
        """
        If a tangent line intersects other circles, try to fix it by adjusting slope
        so that it becomes tangent to one of those intersecting circles. Then
        extend that new line to the original tangent circle.
        """
        for circle in intersecting_circles:
            # ERROR FIX: remove 'self' as the first argument:
            tangent_lines = self.tangent_from_pt_to_circle(tangent["tangent_point_circle1"], circle)
            
            # This 'max' expression presumably tries to pick the line whose slope is "closest" or "farthest" 
            # from the original slope. If you're trying a dot product approach, it might be:
            #   abs(a1*a2 + b1*b2).
            # The original had an odd nested abs. We'll keep the basic idea but fix the redundancy:
            tangent_line = max(
                tangent_lines,
                key=lambda x: abs(
                    x["line_eq"]["a"]*tangent["line_eq"]["a"] 
                    + x["line_eq"]["b"]*tangent["line_eq"]["b"]
                )
            )
            # print(tangent_line)
            # If this new line also intersects the original tangent_circle, we fix its point
            line = {"a": tangent_line["line_eq"]["a"], "b": tangent_line["line_eq"]["a"], "c": tangent_line["line_eq"]["c"]}
            if not self.tangent_intersection(line, tangent_circle):
                # print("intersecting")
                intersecting_point = self.line_circle_intersection_points(tangent_line, tangent_circle, tangent["point"])
                # If there's an intersection, place the new "start" at that intersection:
                if intersecting_point is not None:
                    tangent_line["tangent_point_circle2"] = intersecting_point
                return tangent_line
            
        return []
    
    def tangent_from_multiple_points_to_circles(self, points):
        """
        Generates tangent lines from 'point' to every circle in self.Map_qz,
        then checks intersections and possibly adjusts them.
        """
        tangents_and_circles_list = []
        for point in points:
            # print(f"points: {point}")
            for circle in self.Map_qz:
                tangent_lines = self.tangent_from_pt_to_circle(point, circle)
                # print(f"circle: {circle} and tangent: {tangent_lines}")

                for i, tangent in enumerate(tangent_lines):
                    line = {"a": tangent["line_eq"]["a"], "b": tangent["line_eq"]["b"], "c": tangent["line_eq"]["c"]}
                    intersecting_circles = self.tangent_intersection(line, circle)
                    # print(f"intersecting_circles: {intersecting_circles}")
                    if intersecting_circles:
                        new_tangent_line = self.tangent_no_intersection(intersecting_circles, tangent, circle)
                    else:
                        new_tangent_line = tangent
                    tangent_lines[i] = new_tangent_line
                # print(f"circle: {circle} and tangent: {tangent_lines}")
                if all(len(tangent_line) != 0 for tangent_line in tangent_lines):
                    tangents_and_circles_list.append(tangent_lines)
        
        return tangents_and_circles_list
    
    def common_tangents(self, circle1, circle2, tol=1e-9):
        """
        Given two circles circle1 and circle2,
        each specified as (x, y, r),
        return a list of dictionaries.
        
        Each dictionary represents one common tangent and contains:
        - "line_eq": a dict with slope m and intercept c (line: y = m*x + c)
        - "tangent_point_circle1": point of tangency on circle1
        - "tangent_point_circle2": point of tangency on circle2
        """
        (x1, y1, r1) = circle1
        (x2, y2, r2) = circle2
        Δx = x2 - x1
        Δy = y2 - y1

        tangents = []
        # We try the four sign combinations.
        # For direct tangents use (s1, s2) = (1,1) and (-1,-1).
        # For transverse tangents use (1,-1) and (-1,1).
        for s1, s2 in [(1,1), (-1,-1), (1,-1), (-1,1)]:
            D = s2 * r2 - s1 * r1  # difference term in the tangent condition
            # The equation:
            #   m * Δx - Δy = D * sqrt(1+m^2)
            # Square both sides:
            #   (m*Δx - Δy)^2 = D^2 * (1+m^2)
            # Expand:
            #   m^2*Δx^2 - 2*m*Δx*Δy + Δy^2 = D^2 + D^2*m^2
            # Rearranged as a quadratic in m:
            #   m^2*(Δx^2 - D^2) - 2*m*Δx*Δy + (Δy^2 - D^2) = 0
            A_coef = Δx**2 - D**2
            B_coef = -2 * Δx * Δy
            C_coef = Δy**2 - D**2

            # If A_coef is near zero, skip to avoid division by zero.
            if abs(A_coef) < tol:
                continue

            discriminant = B_coef**2 - 4 * A_coef * C_coef
            if discriminant < -tol:
                continue  # no real solution for m
            elif discriminant < 0:
                discriminant = 0  # numerical fix

            sqrt_disc = sympy.sqrt(discriminant)

            # There may be up to two solutions for m.
            for sign in [1, -1]:
                m = (-B_coef + sign * sqrt_disc) / (2 * A_coef)
                # Compute c using circle1's tangency condition:
                c = y1 - m * x1 + s1 * r1 * sympy.sqrt(1 + m*m)
                
                # The tangent line is: y = m*x + c.
                # Compute the foot of the perpendicular (the tangency point) on each circle.
                def foot_of_perp(circle1, circle2):
                    x1, y1, r1 = circle1
                    x2, y2, r2 = circle2
                    # For line: y = m*x + c, the foot of the perpendicular from (x0, y0) is:
                    x_t1 = (x1 + m*(y1- c)) / (1 + m*m)
                    y_t1 = (m*x1 + m*m*y1 + c) / (1 + m*m)
                    x_t2 = (x2 + m*(y2- c)) / (1 + m*m)
                    y_t2 = (m*x2 + m*m*y2 + c) / (1 + m*m)
                    
                    radius_1 = sympy.sqrt( (x_t1-x1)**2 + (y_t1-y1)**2)
                    radius_2 = sympy.sqrt( (x_t2-x2)**2 + (y_t2-y2)**2)
                    
                    # print(radius_1, radius_2)
                    if not (0.99*r1 <= radius_1 and radius_1 <= 1.01*r1 and 0.99*r2 <= radius_2  and radius_2 <= 1.01*r2):
                        # print("entered")
                        return (None, None), (None, None)
                    
                    return (x_t1, y_t1), (x_t2, y_t2)
                
                T1, T2 = foot_of_perp(circle1, circle2)
                
                if not (T1 == (None,None) and T2 == (None, None)):
                    tangent_data = {
                        "line_eq": {"a": m, "b": -1, "c": c},
                        "tangent_point_circle1": T1,
                        "tangent_point_circle2": T2,
                        "circle1": circle1,
                        "circle2": circle2,
                        "tangent_type": "direct" if s1 == s2 else "transverse"
                    }
                    tangents.append(tangent_data)
        
        return tangents

### convert to parametetic form, i.e. theta
### Assume anticlockwise rotation like in normal axis
#   y
#   ↑
#   |
#   |         
#   |__________→ x

def xy_to_parametric_form(circle, point):
    x,y,r = circle[0], circle[1], circle[2]
    # print((point[1]-y)/(point[0]-x))
    theta = sympy.simplify(sympy.atan2((point[1]-y),(point[0]-x)))
    theta = theta if theta>= 0 else theta + 2 * sympy.pi
    # print(theta)
    return theta