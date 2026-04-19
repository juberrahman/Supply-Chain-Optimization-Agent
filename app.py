import streamlit as st
import duckdb
import pandas as pd
import sys
import os

# Ensure local modules (risk_agent.py, optimizer.py) are recognized
sys.path.append(os.path.dirname(__file__))

from risk_agent import RiskAgent
from optimizer import run_optimization

# --- PAGE CONFIG ---
st.set_page_config(page_title="Agentic Supply Chain Optimizer", layout="wide")
st.title("🤖 Agentic Supply Chain Optimizer")

# --- DATA LOADING (Sanitized for DuckDB) ---
@st.cache_data
def load_all_data():
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
            df = pd.read_csv(path, dtype=str) 
            df.columns = df.columns.str.strip()
            for col in df.columns:
                # Remove common data formatting issues
                df[col] = df[col].str.strip().str.replace(',', '', regex=False).str.replace('$', '', regex=False)
                numeric_cols = ['Weight', 'Unit quantity', 'minm_wgh_qty', 'max_wgh_qty', 
                               'Daily Capacity', 'Cost/unit', 'minimum cost', 'rate']
                if col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
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

# --- SIDEBAR ---
with st.sidebar:
    st.header("Disruption Source")
    # Using a key for file uploader to maintain state
    event_file = st.file_uploader("Upload Events (CSV/JSON)", type=['csv', 'json'], key="event_uploader")
    st.divider()
    st.info("Agent Status: Active (Discovery Mode)")

# --- MAIN UI ---
# Sort IDs to ensure consistent index mapping in the selectbox
order_ids = sorted(con.execute("SELECT DISTINCT \"Order ID\" FROM orders").df()["Order ID"].tolist())

# Added key="selected_order_id" to prevent the app from resetting the choice on rerun
selected_id = st.selectbox("🎯 Select Order ID to Route", order_ids, key="selected_order_id")

if selected_id:
    # 1. Get primary order info
    order_info = con.execute(f"SELECT * FROM orders WHERE \"Order ID\" = {selected_id}").df().iloc[0]
    
    # 2. RAG CONTEXT: Fetch routing facts for the AI
    db_context_df = con.execute(f"""
        SELECT pm."Plant Code", pp.Port, f.Carrier
        FROM product_mapping pm
        JOIN plant_ports pp ON pm."Plant Code" = pp."Plant Code"
        JOIN freight f ON pp.Port = f.orig_port_cd
        WHERE pm."Product ID" = {order_info["Product ID"]}
        AND f.dest_port_cd = '{order_info["Destination Port"]}'
        LIMIT 1
    """).df()
    
    # Safety check for empty database results
    if not db_context_df.empty:
        routing_context = db_context_df.to_dict('records')
    else:
        routing_context = [{"Plant Code": "N/A", "Port": "N/A", "Carrier": "N/A"}]

    st.subheader(f"Current Order: {selected_id}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Product ID", order_info["Product ID"])
    m2.metric("Destination", order_info["Destination Port"])
    m3.metric("Qty", int(order_info["Unit quantity"]))
    m4.metric("Weight", f"{order_info['Weight']} kg")

    # --- SQL FOR OPTIMIZATION ---
    base_query = f"""
    SELECT 
        pm."Plant Code" as plant, pp.Port as origin_port, f.Carrier, f.mode_dsc,
        c."Daily Capacity" as capacity, wh."Cost/unit" as wh_cost,
        f."minimum cost" as min_f, f.rate as rate_f
    FROM product_mapping pm
    JOIN plant_ports pp ON pm."Plant Code" = pp."Plant Code"
    JOIN freight f ON pp.Port = f.orig_port_cd
    JOIN capacity c ON pm."Plant Code" = c."Plant ID"
    JOIN wh_costs wh ON pm."Plant Code" = wh.WH
    WHERE pm."Product ID" = {order_info["Product ID"]}
      AND f.dest_port_cd = '{order_info["Destination Port"]}'
      AND {order_info["Weight"]} BETWEEN f.minm_wgh_qty AND f.max_wgh_qty
    """
    baseline_routes = con.execute(base_query).df().drop_duplicates()

    col_left, col_right = st.columns(2)

    # LEFT COLUMN: BASELINE
    with col_left:
        st.subheader("⚪ Baseline Routing")
        if baseline_routes.empty:
            st.warning("No physical routes found.")
        else:
            winners, cost, viol, msg = run_optimization(baseline_routes.copy(), order_info)
            if winners:
                st.metric("Standard Total Cost", f"${cost:,.2f}")
                for r in winners:
                    st.write(f"✅ **{r['plant']}** via **{r['Carrier']}** ({int(r['assigned_qty'])} units)")
                if viol: st.warning("⚠️ Capacity exceeded in baseline.")
            else:
                st.error(msg)

    # RIGHT COLUMN: AGENTIC
    with col_right:
        st.subheader("🔴 Agentic Routing (AI-Adjusted)")
        if event_file:
            # --- INITIALIZE AGENT DATA ---
            events_df = pd.read_csv(event_file) if 'csv' in event_file.name else pd.read_json(event_file)
            agent = RiskAgent()
            full_text = " ".join(events_df['Description'].astype(str).tolist())

            # --- STEP A: REASONING SUMMARY ---
            with st.spinner("Gemini is checking database facts..."):
                summary = agent.summarize_risks(full_text, selected_id, routing_context)
                st.info(f"**AI Risk Analysis:**\n\n{summary}")

            # --- STEP B: AI MAPPING ---
            with st.spinner("Extracting multipliers..."):
                disruption_json = agent.parse_events(full_text)
            
            st.write("Review AI Multipliers:")
            # key="risk_editor" ensures the table doesn't refresh unexpectedly
            reviewed_disruptions = st.data_editor(disruption_json, num_rows="dynamic", key="risk_editor")
            
            if st.button("Apply AI Constraints & Re-Route", key="apply_reroute_btn"):
                # 1. Start with a fresh copy and force Float types to prevent TypeError
                adjusted_df = baseline_routes.copy()
                for col_name in ['capacity', 'rate_f', 'wh_cost']:
                    adjusted_df[col_name] = pd.to_numeric(adjusted_df[col_name], errors='coerce').astype(float)
                
                # 2. APPLY MULTIPLIERS
                # Convert the editor result to a DataFrame for stable iteration
                for _, disruption in pd.DataFrame(reviewed_disruptions).iterrows():
                    target_id = str(disruption['target_id']).strip()
                    cap_mult = float(disruption.get('capacity_multiplier', 1.0))
                    cost_mult = float(disruption.get('cost_multiplier', 1.0))
                    
                    if disruption['target_type'] == 'plant':
                        mask = adjusted_df['plant'] == target_id
                        adjusted_df.loc[mask, 'capacity'] *= cap_mult
                        adjusted_df.loc[mask, 'wh_cost'] *= cost_mult
                    
                    elif disruption['target_type'] == 'port':
                        mask = adjusted_df['origin_port'] == target_id
                        adjusted_df.loc[mask, 'capacity'] *= cap_mult
                        adjusted_df.loc[mask, 'rate_f'] *= cost_mult
                    
                    elif disruption['target_type'] == 'carrier':
                        mask = adjusted_df['Carrier'] == target_id
                        adjusted_df.loc[mask, 'rate_f'] *= cost_mult

                # 3. RUN OPTIMIZATION
                winners_a, cost_a, viol_a, msg_a = run_optimization(adjusted_df, order_info)
                
                if winners_a:
                    delta = cost_a - cost
                    st.metric("Adjusted Total Cost", f"${cost_a:,.2f}", delta=f"${delta:,.2f}", delta_color="inverse")
                    for ra in winners_a:
                        st.write(f"🚀 **{ra['plant']}** via **{ra['Carrier']}** ({int(ra['assigned_qty'])} units)")
                else:
                    # Provide helpful context for feasibility failure
                    total_cap = adjusted_df['capacity'].sum()
                    st.error(f"Fulfillment Infeasible. Total Capacity: {total_cap} | Required: {order_info['Unit quantity']}")
        else:
            st.info("Upload `events.csv` to activate Agentic routing.")

st.divider()
st.caption("Supply Chain Agent v2.8 | Stable State & Multi-Type Adjustment Enabled")