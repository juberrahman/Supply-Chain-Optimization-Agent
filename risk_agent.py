import streamlit as st
import google.generativeai as genai
import re
import json
import os

class RiskAgent:
    def __init__(self):
        """
        Initializes the agent by fetching the API key from Streamlit Secrets 
        and discovering the best available Gemini model.
        """
        self.model = None
        try:
            # 1. Fetch API Key from Streamlit Secrets (Production) or OS Environment (Local)
            if "GEMINI_API_KEY" in st.secrets:
                self.api_key = st.secrets["GEMINI_API_KEY"]
            else:
                self.api_key = os.getenv("GEMINI_API_KEY")
            
            if not self.api_key:
                st.error("🔑 API Key not found! Add GEMINI_API_KEY to Streamlit Secrets.")
                return

            # 2. Configure Google AI
            genai.configure(api_key=self.api_key)
            
            # 3. Discover the best model (Stable Discovery Pattern)
            self.model = self._get_latest_model()
            
        except Exception as e:
            st.error(f"RiskAgent Initialization Error: {e}")

    def _get_latest_model(self):
        """Finds the newest available Gemini model on the account."""
        try:
            models = genai.list_models()
            candidates = []

            for m in models:
                if "generateContent" in m.supported_generation_methods:
                    name = m.name.replace("models/", "")
                    if "gemini" in name.lower():
                        candidates.append(name)

            if not candidates:
                raise Exception("No Gemini models available for this key.")

            # Sort by version (e.g., 1.5 > 1.0)
            def version_key(name):
                nums = re.findall(r"\d+(?:\.\d+)?", name)
                return [float(x) for x in nums] if nums else [0]
            
            candidates.sort(key=version_key, reverse=True)

            # Test the best candidate
            for model_name in candidates:
                try:
                    model = genai.GenerativeModel(model_name)
                    model.generate_content("ping") 
                    return model
                except:
                    continue
            
            raise Exception("No responding models found.")
        except Exception as e:
            raise Exception(f"Model discovery failed: {e}")

    def summarize_risks(self, event_text, order_id, db_context):
        """
        Uses RAG context to explain news impacts on a specific order.
        """
        if not self.model:
            return "⚠️ AI Summary unavailable (Model not initialized)."

        # Safety check for database context
        if not db_context or not isinstance(db_context, list):
            return "No routing facts found in database for this order."

        ctx = db_context[0]
        details = (f"Order {order_id} uses Plant {ctx.get('Plant Code', 'N/A')}, "
                   f"Port {ctx.get('Port', 'N/A')}, and Carrier {ctx.get('Carrier', 'N/A')}.")
        
        prompt = f"""
        FACTS: {details}
        NEWS: "{event_text}"

        TASK: Explain how the news impacts this specific order based on the facts provided.
        Be concise (max 3 sentences). Do not ask questions.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Summary failed: {e}"

    def parse_events(self, event_text):
        """
        Extracts structured multipliers from news for the optimizer.
        """
        if not self.model:
            return []

        prompt = f"""
        Analyze news and return ONLY a JSON list of impacts.
        IDs: PLANT01-16, PORT01-11, V444_0-8.
        
        Logic:
        - 'Northeast' or 'Fire' -> PLANT16
        - 'Strike' or 'Port' -> PORT09
        
        Format:
        [{{
            "target_type": "plant" | "port" | "carrier",
            "target_id": "STRING",
            "status": "closed" | "partial",
            "capacity_multiplier": float (0.0 to 1.0),
            "cost_multiplier": float (1.0+),
            "description": "Short reasoning"
        }}]
        
        News: {event_text}
        """
        try:
            response = self.model.generate_content(prompt)
            raw_json = response.text.strip().replace('```json', '').replace('```', '')
            return json.loads(raw_json)
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return []