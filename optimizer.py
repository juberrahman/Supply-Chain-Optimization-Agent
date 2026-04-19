from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpStatus

def run_optimization(routes_df, order_info, disruptions=None):
    df = routes_df.copy()
    
    # ... (Keep your disruption logic here) ...

    prob = LpProblem("Multi_Plant_Optimization", LpMinimize)
    
    # CHANGE: Use Continuous variables instead of Binary
    # This represents the QUANTITY sent from each route
    route_vars = LpVariable.dicts("Qty", df.index, lowBound=0, cat="Continuous")
    
    # OBJECTIVE: Min(Sum of (Wh Cost + Rate) * Qty)
    # Note: This simplifies freight to a per-unit basis for the split
    costs = {}
    for i, row in df.iterrows():
        # Estimate per-unit freight (Rate + Min divided by avg qty)
        per_unit_f = row['rate_f'] + (row['min_f'] / order_info['Unit quantity'])
        costs[i] = row['wh_cost'] + per_unit_f
    
    prob += lpSum([route_vars[i] * costs[i] for i in df.index])
    
    # CONSTRAINT 1: Total Qty from all plants must EQUAL Order Qty
    prob += lpSum([route_vars[i] for i in df.index]) == order_info['Unit quantity']
    
    # CONSTRAINT 2: Qty from each plant cannot exceed its Capacity
    for i, row in df.iterrows():
        prob += route_vars[i] <= row['capacity']
            
    prob.solve()
    
    if LpStatus[prob.status] == 'Optimal':
        # Find all routes that were assigned quantity > 0
        selected_routes = []
        for i in df.index:
            if value(route_vars[i]) > 0:
                res = df.loc[i].to_dict()
                res['assigned_qty'] = value(route_vars[i])
                selected_routes.append(res)
        
        return selected_routes, value(prob.objective), False, "Success"
    else:
        return None, 0, False, "Infeasible: Total capacity of all plants is too low."