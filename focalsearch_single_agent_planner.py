import heapq
import time

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0,0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def move_joint_state(locs, dir):

    new_locs = []
    for i, dir in enumerate(dir):
        x, y = locs[i]
        new_locs.append((x + dir[0], y + dir[1]))
    return new_locs

def generate_motions_recursive(num_agents,cur_agent, agent_motions = []):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    if cur_agent == num_agents:
        return [agent_motions]
    joint_state_motions = []
    for direction in directions:
        next_agent_motions = generate_motions_recursive(num_agents, cur_agent + 1, agent_motions + [direction])
        joint_state_motions.extend(next_agent_motions)

    return joint_state_motions 


def is_valid_motion(old_loc, new_loc):
    ##############################
    # Task 1.3/1.4: Check if a move from old_loc to new_loc is valid
    # Check if two agents are in the same location (vertex collision)
    #  
    if len(set(old_loc)) != len(old_loc):
        return False
    
    # Check edge collision
    #  
    for i in range(len(old_loc)):
        for j in range(i+1, len(new_loc)):
            if new_loc[i] == old_loc[j] and new_loc[j] == old_loc[i]:
                return False
    return True

def get_sum_of_cost(paths):
    rst = 0
    if paths is None:
        return -1
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3/1.4: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    constraint_table = [[]]
    for constraint in constraints:
        if constraint['agent'] == agent:
            timestep = constraint['timestep']
            while len(constraint_table)<=timestep:
                constraint_table.append([])
            constraint_table[timestep].append(constraint['loc'])
    return constraint_table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3/1.4: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.
    if next_time<len(constraint_table):
        for constraint in constraint_table[next_time]:
            if (len(constraint) == 1 and next_loc == constraint[0]) or (len(constraint) == 2 and curr_loc == constraint[0] and next_loc == constraint[1]):
                return True

    return False

def is_future_constrained(goal_loc, curr_time, constraint_table):
    last_time = len(constraint_table)
    for next_time in range(curr_time, last_time):
        for constraint in constraint_table[next_time]:
            if goal_loc == constraint[0]:
                return True
    return False

def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node['d_val'], node))

def push_node_focal(focal_list, node):
    heapq.heappush(focal_list, (node['d_val'], node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(focal_list, open_list):
    _, _, _, _, curr = heapq.heappop(focal_list)
    for idx, ( _, _, _, _, a_dict) in enumerate(open_list):
        if a_dict is curr:  
            del open_list[idx]  
            break
    return curr

def create_focal_list(open_list, w):
    if not open_list:  # Check if list A is empty
        return []
    focal_list=[]
    threshold = w * open_list[0][0] 
    for _, tup in enumerate(open_list):
        if tup[0]<= threshold:
            push_node_focal(focal_list, tup[-1])
        else:
            break
    return focal_list

# def pop_node(open_list):
#     _, _, _, curr = heapq.heappop(open_list)
#     return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def in_map(map, loc):
    if loc[0] >= len(map) or loc[1] >= len(map[0]) or min(loc) < 0:
        return False
    else:
        return True

def all_in_map(map, locs):
    for loc in locs:
        if not in_map(map, loc):
            return False
    return True

def get_d_val(timestep, loc, other_paths):
    d_val = 0
    for path in other_paths:
        if len(path)>timestep:
            if loc == path[timestep]:
                d_val+=1
    return d_val

def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints, other_paths = [], w=1.2):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # Task 1.2/1.3/1.4: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.
    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    total_false = sum(value is False for row in my_map for value in row)
    h_value = h_values[start_loc]
    constraint_table = build_constraint_table(constraints, agent)
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'd_val':0, 'parent': None, 'timestep': 0}
    push_node(open_list, root)
    focal_list = create_focal_list(open_list, w)
    closed_list[(root['loc'], root['timestep'])] = root
    while len(open_list) > 0:
        # print('Len',len(focal_list), len(open_list))
        curr = pop_node(focal_list, open_list)
        #############################
        # Task 2.2: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc and not is_future_constrained(goal_loc, curr['timestep'], constraint_table):
            return get_path(curr), min(open_list[0][0], curr['g_val']+curr['h_val'])
        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if not in_map(my_map, child_loc) or my_map[child_loc[0]][child_loc[1]]:
                continue
            if is_constrained(curr['loc'], child_loc, curr['timestep']+1, constraint_table):
                continue
            d_val = get_d_val(curr['timestep']+1, child_loc, other_paths)
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'd_val':d_val,
                    'parent': curr,
                    'timestep': curr['timestep']+1}
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
                    # focal_list = create_focal_list(open_list, w)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)
                # focal_list = create_focal_list(open_list, w)

        if child['timestep']>1*total_false:
            return None
        focal_list = create_focal_list(open_list, w)

    return None  # Failed to find solutions



def joint_state_a_star(my_map, starts, goals, h_values, num_agents):
    """ my_map      - binary obstacle map
        start_loc   - start positions
        goal_loc    - goal positions
        num_agent   - total number of agents in fleet
    """
    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = 0
     ##############################
    # Task 1.1: Iterate through starts and use list of h_values to calculate total h_value for root node
    #  
    h_value = sum(h_values[i][start] for i, start in enumerate(starts))
    root = {'loc': starts, 'g_val': 0, 'h_val': h_value, 'parent': None }
    push_node(open_list, root)
    closed_list[tuple(root['loc'])] = root

     ##############################
    # Task 1.1:  Generate set of all possible motions in joint state space
    #
    #   
    directions = generate_motions_recursive(num_agents,0)
    while len(open_list) > 0:
        curr = pop_node(open_list)
        
        if curr['loc'] == goals:
            return get_path(curr)

        for dir in directions:
            
            ##############################
            # Task 1.1:  Update position of each agent
            #
            #  
            child_loc = move_joint_state(curr['loc'], dir)

            if not all_in_map(my_map, child_loc):
                continue
             ##############################
            # Task 1.1:  Check if any agent is in an obstacle
            #
            valid_move = all(not my_map[x][y] for x, y in child_loc)
            #  
            
            if not valid_move:
                continue

             ##############################
            # Task 1.1:   check for collisions
            #
            #  
            if not is_valid_motion(curr['loc'],child_loc):
                continue
            
             ##############################
            # Task 1.1:  Calculate heuristic value
            #
            #  

            h_value = sum(h_values[i][loc] for i,loc in enumerate(child_loc))

            # Create child node
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + num_agents,
                    'h_val': h_value,
                    'parent': curr}
            if tuple(child['loc']) in closed_list:
                existing_node = closed_list[tuple(child['loc'])]
                if compare_nodes(child, existing_node):
                    closed_list[tuple(child['loc'])] = child
                    push_node(open_list, child)
            else:
                closed_list[tuple(child['loc'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions