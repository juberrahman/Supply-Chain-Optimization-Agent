import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load local .env if it exists (for local dev)
load_dotenv()

class RiskAgent:
    def __init__(self):
        # 1. Try to get key from Streamlit Secrets (Cloud)
        # 2. Fallback to Environment Variables (Local .env)
        self.api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            st.error("🔑 API Key missing! Please check Streamlit Secrets or your .env file.")
            st.stop()
            
        genai.configure(api_key=self.api_key)
        
        try:
            self.model = self._get_latest_model()
        except Exception as e:
            # Provide more detail if the key is invalid
            raise Exception(f"Model discovery failed. Check if your API key is valid. Error: {e}")

    def _get_latest_model(self):
        """Finds the newest available Gemini model on your account."""
        try:
            models = genai.list_models()
            candidates = []

            for m in models:
                if "generateContent" in m.supported_generation_methods:
                    # Strip 'models/' prefix for the constructor
                    name = m.name.replace("models/", "")
                    if "gemini" in name.lower():
                        candidates.append(name)

            if not candidates:
                raise Exception("No Gemini models available for this API key.")

            # Sort by version (1.5 > 1.0)
            def version_key(name):
                nums = re.findall(r"\d+(?:\.\d+)?", name)
                return [float(x) for x in nums] if nums else [0]

            candidates.sort(key=version_key, reverse=True)

            # Try to 'ping' each model until one works
            for model_name in candidates:
                try:
                    model = genai.GenerativeModel(model_name)
                    # We do a tiny test call to verify this specific name works
                    model.generate_content("ping") 
                    print(f"✅ Successfully connected to: {model_name}")
                    return model
                except Exception:
                    continue

            raise Exception("None of the available models responded.")

        except Exception as e:
            raise Exception(f"Model discovery failed: {e}")

    def summarize_risks(self, event_text, order_id, db_context):
        """Now aware of DuckDB data for this specific order."""
        
        # Convert the database row into a readable string for the AI
        details = f"Order {order_id} uses: Plant {db_context[0]['Plant Code']}, Port {db_context[0]['Port']}, and Carrier {db_context[0]['Carrier']}."
        
        prompt = f"""
        CONTEXT FROM DATABASE:
        {details}

        NEWS EVENTS:
        "{event_text}"

        TASK:
        Based on the database context provided, explain exactly how the news impacts this order. 
        Do not ask the user for information. Be concise (3 sentences max).
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Summary failed: {e}"

    def parse_events(self, event_text):
        # We put our custom Supply Chain logic back in here
        prompt = f"""
        Return ONLY valid JSON for disruption data.
        IDs: PLANT01-16, PORT01-11, V444_0-8.
        
        Rules:
        - If 'Northeast' or 'Transformer' mentioned -> target_id: PLANT16
        - If 'Strike' or 'Port' mentioned -> target_id: PORT09
        
        Events: {event_text}
        
        Format:
        [{{
            "target_type": "plant" | "port" | "carrier",
            "target_id": "STRING",
            "status": "closed" | "partial",
            "capacity_multiplier": float,
            "cost_multiplier": float,
            "description": "Short reasoning"
        }}]
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(text)
        except Exception as e:
            print("JSON parse error:", e)
            return []