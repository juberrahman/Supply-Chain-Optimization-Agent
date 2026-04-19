import streamlit as st
import duckdb
import pandas as pd
import sys
import os

# Ensure local modules can be found
sys.path.append(os.path.dirname(__file__))

from risk_agent import RiskAgent
from optimizer import run_optimization

# --- PAGE CONFIG ---
st.set_page_config(page_title="Agentic Supply Chain Optimizer", layout="wide")
st.title("🤖 Agentic Supply Chain Optimizer")

# --- DATA LOADING ---
@st.cache_data
def load_all_data():
    """Loads CSV files and performs initial cleanup."""
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
            df.columns = df.columns.str.strip()
            # Clean only the string columns
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            data[table] = df
        except Exception as e:
            st.error(f"Error reading {path}: {e}")
            st.stop()
    return data

# Initialize Database
db_tables = load_all_data()
con = duckdb.connect(database=':memory:')
for name, df in db_tables.items():
    con.register(name, df)

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("Disruption Source")
    event_file = st.file_uploader("Upload Events (CSV/JSON)", type=['csv', 'json'])
    st.divider()
    st.info("Agent Status: Active (Gemini 1.5 Flash)")

# --- MAIN UI: ORDER SELECTION ---
order_ids = con.execute("SELECT DISTINCT \"Order ID\" FROM orders").df()["Order ID"].tolist()
selected_id = st.selectbox("🎯 Select Order ID to Route", order_ids)

if selected_id:
    # Get specific order info
    order_info = con.execute(f"SELECT * FROM orders WHERE \"Order ID\" = {selected_id}").df().iloc[0]
    
    st.subheader(f"Current Order: {selected_id}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Product ID", order_info["Product ID"])
    m2.metric("Destination", order_info["Destination Port"])
    m3.metric("Qty", int(order_info["Unit quantity"]))
    m4.metric("Weight", f"{order_info['Weight']} kg")

    # --- THE "OLD TRICK" SQL QUERY ---
    # We use REPLACE inside the SQL to strip commas and symbols during the JOIN
    base_query = f"""
    SELECT 
        pm."Plant Code" as plant, 
        pp.Port as origin_port, 
        f.Carrier, 
        f.mode_dsc,
        CAST(REPLACE(CAST(c."Daily Capacity" AS VARCHAR), ',', '') AS DOUBLE) as capacity,
        CAST(REPLACE(CAST(wh."Cost/unit" AS VARCHAR), ',', '') AS DOUBLE) as wh_cost,
        CAST(REPLACE(REPLACE(REPLACE(CAST(f."minimum cost" AS VARCHAR), '$', ''), ',', ''), ' ', '') AS DOUBLE) as min_f,
        CAST(REPLACE(REPLACE(REPLACE(CAST(f.rate AS VARCHAR), '$', ''), ',', ''), ' ', '') AS DOUBLE) as rate_f
    FROM product_mapping pm
    JOIN plant_ports pp ON pm."Plant Code" = pp."Plant Code"
    JOIN freight f ON pp.Port = f.orig_port_cd
    JOIN capacity c ON pm."Plant Code" = c."Plant ID"
    JOIN wh_costs wh ON pm."Plant Code" = wh.WH
    WHERE pm."Product ID" = {order_info["Product ID"]}
      AND f.dest_port_cd = '{order_info["Destination Port"]}'
      AND CAST({order_info["Weight"]} AS DOUBLE) BETWEEN 
          CAST(REPLACE(CAST(f.minm_wgh_qty AS VARCHAR), ',', '') AS DOUBLE) AND 
          CAST(REPLACE(CAST(f.max_wgh_qty AS VARCHAR), ',', '') AS DOUBLE)
    """
    
    try:
        baseline_routes = con.execute(base_query).df().drop_duplicates()
    except Exception as e:
        st.error(f"Database Engine Error: {e}")
        st.stop()

    # --- COMPARISON VIEW ---
    col_left, col_right = st.columns(2)

    # 1. BASELINE ROUTING
    with col_left:
        st.subheader("⚪ Baseline Routing")
        if baseline_routes.empty:
            st.warning("No physical routes found for this weight/destination.")
        else:
            winner, cost, viol, msg = run_optimization(baseline_routes.copy(), order_info)
            if winner is not None:
                st.metric("Standard Cost", f"${cost:,.2f}")
                st.write(f"**Plant:** {winner['plant']} | **Carrier:** {winner['Carrier']}")
                if viol: st.warning("Note: Exceeds standard capacity.")
            else:
                st.error(msg)

    # 2. AGENTIC ROUTING
    with col_right:
        st.subheader("🔴 Agentic Routing (AI-Adjusted)")
        if event_file:
            events_df = pd.read_csv(event_file) if 'csv' in event_file.name else pd.read_json(event_file)
            agent = RiskAgent()
            
            with st.spinner("Gemini is reading news/emails..."):
                disruption_json = agent.parse_events(events_df.to_string())
            
            st.write("AI Disruption Assessment (Editable):")
            reviewed_disruptions = st.data_editor(disruption_json, num_rows="dynamic")
            
            if st.button("Apply AI Constraints & Re-Route"):
                winner_a, cost_a, viol_a, msg_a = run_optimization(baseline_routes.copy(), order_info, reviewed_disruptions)
                
                if winner_a is not None:
                    delta = cost_a - cost
                    st.metric("Adjusted Cost", f"${cost_a:,.2f}", delta=f"${delta:,.2f}", delta_color="inverse")
                    st.write(f"**New Plant:** {winner_a['plant']} | **Carrier:** {winner_a['Carrier']}")
                    if viol_a: st.warning("Note: Capacity shortfall persists.")
                else:
                    st.error("AI-selected path is completely blocked.")
        else:
            st.info("Upload an `events.csv` to activate Agentic routing logic.")

st.divider()
st.caption("Supply Chain Agent v2.2 | Logic: SQL REPLACE Tricks + PuLP Optimization")