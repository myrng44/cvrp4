import sys
import numpy as np

def transform(input_file, output_file, filo_sol_file=None):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    capacity = 0
    node_coords = {}
    demands = {}
    depot_node = 1
    
    section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('CAPACITY'):
            capacity = float(line.split(':')[1].strip())
            continue
        elif line.startswith('NODE_COORD_SECTION'):
            section = 'COORD'
            continue
        elif line.startswith('DEMAND_SECTION'):
            section = 'DEMAND'
            continue
        elif line.startswith('DEPOT_SECTION'):
            section = 'DEPOT'
            continue
        elif line.startswith('EOF') or line.startswith('EDGE_WEIGHT_TYPE') or line.startswith('NAME') or line.startswith('TYPE') or line.startswith('DIMENSION') or line.startswith('COMMENT'):
            if line.startswith('EOF'):
                section = None
            continue
            
        if section == 'COORD':
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node_coords[node_id] = (x, y)
        elif section == 'DEMAND':
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                demand = int(parts[1])
                demands[node_id] = demand
        elif section == 'DEPOT':
            parts = line.split()
            if len(parts) >= 1:
                val = int(parts[0])
                if val != -1:
                    depot_node = val
                    section = None

    nodes = sorted(list(node_coords.keys()))
    depot_x, depot_y = node_coords[depot_node]
    
    customers = [n for n in nodes if n != depot_node]
    customer_coords = [node_coords[n] for n in customers]
    customer_demands = [demands[n] for n in customers]
    
    all_x = [c[0] for c in node_coords.values()]
    all_y = [c[1] for c in node_coords.values()]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Scale max value so the coords fall into [0,1]
    scale_x = max_x - min_x if max_x > min_x else 1.0
    scale_y = max_y - min_y if max_y > min_y else 1.0
    scale = max(scale_x, scale_y)
    
    def norm_x(x): return (x - min_x) / scale
    def norm_y(y): return (y - min_y) / scale
    
    out_parts = []
    
    # depot
    out_parts.append('depot')
    out_parts.append(str(norm_x(depot_x)))
    out_parts.append(str(norm_y(depot_y)))
    
    # customers
    out_parts.append('customer')
    for (x, y) in customer_coords:
        out_parts.append(str(norm_x(x)))
        out_parts.append(str(norm_y(y)))
        
    # capacity
    out_parts.append('capacity')
    out_parts.append(str(capacity))
    
    # demand
    out_parts.append('demand')
    for d in customer_demands:
        out_parts.append(str(d))
        
    # cost (scaled optimal from filo)
    out_parts.append('cost')
    out_parts.append('0.0') # Dummy cost since LEHD bypasses it and re-calculates via node_flags
        
    # Read FILO solution if provided
    node_seq = []
    flag_seq = []
    if filo_sol_file:
        try:
            with open(filo_sol_file, 'r', encoding='utf-8') as fsol:
                for sol_line in fsol:
                    if sol_line.startswith('Route #'):
                        r_nodes = [int(x) for x in sol_line.split(':')[1].split()]
                        node_seq.extend(r_nodes)
                        flag_seq.append(1)
                        flag_seq.extend([0] * (len(r_nodes) - 1))
            print(f"Loaded FILO solution: {len(node_seq)} total nodes routed.")
        except Exception as e:
            print(f"Warning: Could not read sol file correctly. Error: {e}")
            
    if len(node_seq) != len(customers):
        print(f"Fallback to dummy node_flag (mismatch/none). Expected {len(customers)}, got {len(node_seq)}")
        node_seq = list(range(1, len(customers) + 1))
        flag_seq = [1] + [0] * (len(customers) - 1)
        
    out_parts.append('node_flag')
    for n in node_seq:
        out_parts.append(str(n))
    for f in flag_seq:
        out_parts.append(str(f))
        
    with open(output_file, 'w') as f:
        f.write(','.join(out_parts) + '\n')
        
    print(f"Data transformed successfully. Length of customers: {len(customers)}")

if __name__ == '__main__':
    input_f = r'f:\PJ\VRP\models\NCO_code\single_objective\LEHD\CVRP\Transform_data\another_data\Antwerp1.txt'
    output_f = r'f:\PJ\VRP\models\NCO_code\single_objective\LEHD\CVRP\Transform_data\another_data\Antwerp1_lkh.txt'
    sol_f = r'f:\PJ\VRP\models\NCO_code\single_objective\LEHD\CVRP\result\filo2\Antwerp1.txt_seed-0.vrp.sol'
    # Trích xuất dữ liệu route từ Filo .sol và ghi vào Antwerp1_lkh.txt
    transform(input_f, output_f, filo_sol_file=sol_f)
