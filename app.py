import streamlit as st
import duckdb
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpStatus

# --- PAGE CONFIG ---
st.set_page_config(page_title="Supply Chain Super Agent", layout="wide")
st.title("📦 Supply Chain Optimization Agent")

# --- DATA LOADING ---
@st.cache_data
def get_all_data():
    """Loads CSV files and ensures numeric columns are correctly typed."""
    files = {
        "orders": "data/OrderList.csv",
        "freight": "data/FreightRates.csv",
        "capacity": "data/WhCapacities.csv",
        "plant_ports": "data/PlantPorts.csv",
        "product_mapping": "data/ProductsPerPlant.csv",
        "wh_costs": "data/WhCosts.csv"
    }
    
    data = {}
    for table, path in files.items():
        try:
            df = pd.read_csv(path)
            # Clean headers: remove trailing/leading spaces
            df.columns = df.columns.str.strip()
            # Clean string values
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            
            # Pre-emptive numeric conversion
            if 'Weight' in df.columns: df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
            if 'Unit quantity' in df.columns: df['Unit quantity'] = pd.to_numeric(df['Unit quantity'], errors='coerce')
            if 'minm_wgh_qty' in df.columns: df['minm_wgh_qty'] = pd.to_numeric(df['minm_wgh_qty'], errors='coerce')
            if 'max_wgh_qty' in df.columns: df['max_wgh_qty'] = pd.to_numeric(df['max_wgh_qty'], errors='coerce')
            if 'Daily Capacity' in df.columns: df['Daily Capacity'] = pd.to_numeric(df['Daily Capacity'], errors='coerce')
            if 'Cost/unit' in df.columns: df['Cost/unit'] = pd.to_numeric(df['Cost/unit'], errors='coerce')
            
            data[table] = df
        except Exception as e:
            st.error(f"Error reading {path}: {e}")
            st.stop()
    return data

# Initialize Database
all_data = get_all_data()
con = duckdb.connect(database=':memory:')
for table_name, df in all_data.items():
    con.register(table_name, df)

# --- SIDEBAR: ORDER SELECTION ---
st.sidebar.header("Order Management")
order_list = con.execute("SELECT DISTINCT \"Order ID\" FROM orders").df()["Order ID"].tolist()
selected_order_id = st.sidebar.selectbox("Select an Order ID", order_list)

if selected_order_id:
    # 1. Fetch Order Details
    order_info = con.execute(f"SELECT * FROM orders WHERE \"Order ID\" = {selected_order_id}").df().iloc[0]
    
    st.subheader(f"Current Order: {selected_order_id}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Product ID", order_info["Product ID"])
    m2.metric("Destination", order_info["Destination Port"])
    m3.metric("Qty", int(order_info["Unit quantity"]))
    m4.metric("Weight", f"{order_info['Weight']} kg")

    # 2. SQL JOIN: Find Physical Paths
    feasible_query = f"""
    SELECT 
        pm."Plant Code" as plant,
        pp.Port as origin_port,
        f.Carrier,
        f.mode_dsc,
        CAST(c."Daily Capacity" AS DOUBLE) as capacity,
        CAST(wh."Cost/unit" AS DOUBLE) as wh_cost,
        CAST(REPLACE(REPLACE(f."minimum cost", '$', ''), ' ', '') AS DOUBLE) as min_f,
        CAST(REPLACE(REPLACE(f.rate, '$', ''), ' ', '') AS DOUBLE) as rate_f
    FROM product_mapping pm
    JOIN plant_ports pp ON pm."Plant Code" = pp."Plant Code"
    JOIN freight f ON pp.Port = f.orig_port_cd
    JOIN capacity c ON pm."Plant Code" = c."Plant ID"
    JOIN wh_costs wh ON pm."Plant Code" = wh.WH
    WHERE pm."Product ID" = {order_info["Product ID"]}
      AND f.dest_port_cd = '{order_info["Destination Port"]}'
      AND CAST({order_info["Weight"]} AS DOUBLE) BETWEEN CAST(f.minm_wgh_qty AS DOUBLE) AND CAST(f.max_wgh_qty AS DOUBLE)
    """
    
    # We define routes_df here before checking if it's empty
    routes_df = con.execute(feasible_query).df().drop_duplicates()

    if routes_df.empty:
        st.warning(f"No feasible route for weight {order_info['Weight']} to {order_info['Destination Port']}. Check weight brackets in FreightRates.csv.")
    else:
        st.write("### Eligible Supply Chain Paths")
        st.dataframe(routes_df, use_container_width=True)
        
        # --- PULP SOLVER ---
        prob = LpProblem("Route_Optimization", LpMinimize)
        route_vars = LpVariable.dicts("Route", routes_df.index, cat="Binary")
        
        # OBJECTIVE: Min(Warehouse Cost + Freight Cost)
        route_costs = []
        for i, row in routes_df.iterrows():
            f_total = max(row['min_f'], row['rate_f'] * order_info['Weight'])
            w_total = row['wh_cost'] * order_info['Unit quantity']
            route_costs.append(f_total + w_total)
        
        prob += lpSum([route_vars[i] * route_costs[i] for i in routes_df.index])
        prob += lpSum([route_vars[i] for i in routes_df.index]) == 1 
        
        # Attempt Capacity Constraint
        for i, row in routes_df.iterrows():
            prob += route_vars[i] * order_info['Unit quantity'] <= row['capacity']
            
        prob.solve()
        
        # Smart Logic: If Capacity fails, solve for Cost only but warn the user
        is_violation = False
        if LpStatus[prob.status] != 'Optimal':
            is_violation = True
            # Solve again without capacity constraint
            prob = LpProblem("Min_Cost_Only", LpMinimize)
            prob += lpSum([route_vars[i] * route_costs[i] for i in routes_df.index])
            prob += lpSum([route_vars[i] for i in routes_df.index]) == 1
            prob.solve()
        
        if LpStatus[prob.status] == 'Optimal':
            best_idx = [i for i in routes_df.index if value(route_vars[i]) == 1][0]
            winner = routes_df.iloc[best_idx]
            
            if is_violation:
                st.warning(f"⚠️ **Capacity Shortfall:** Cheapest route shown, but order exceeds {winner['plant']} capacity.")
            else:
                st.success(f"✨ Optimal Route Found: **{winner['plant']}** via **{winner['origin_port']}**")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Carrier Info**")
                st.write(f"Carrier: {winner['Carrier']}")
                st.write(f"Service: {winner['mode_dsc']}")
            with c2:
                st.write("**Cost Breakdown**")
                st.write(f"Warehouse: ${winner['wh_cost'] * order_info['Unit quantity']:,.2f}")
                st.write(f"Freight: ${max(winner['min_f'], winner['rate_f'] * order_info['Weight']):,.2f}")
            with c3:
                st.metric("Total Cost", f"${value(prob.objective):,.2f}")
        else:
            st.error("Error calculating routes.")

st.divider()
st.caption("Agent Status: Online | Logic: SQL Joins + PuLP Optimization")