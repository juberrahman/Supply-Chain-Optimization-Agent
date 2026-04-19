import streamlit as st
import google.generativeai as genai
import os
import re

# include changed file


class RiskAgent:
    def __init__(self):
        try:
            self.api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

            if not self.api_key:
                raise Exception("API key not found.")

            genai.configure(api_key=self.api_key)
            self.model = self._get_latest_model()

        except Exception as e:
            st.error(f"Authentication Error: {e}")
            raise

    def _get_latest_model(self):
        """Find newest usable Gemini model dynamically."""
        models = genai.list_models()

        candidates = []
        for m in models:
            methods = getattr(m, "supported_generation_methods", [])
            if "generateContent" in methods and "gemini" in m.name.lower():
                model_name = m.name.replace("models/", "")
                candidates.append(model_name)

        if not candidates:
            raise Exception("No Gemini models available.")

        def version_key(name):
            nums = re.findall(r"\d+(?:\.\d+)?", name)
            return [float(n) for n in nums] if nums else [0]

        candidates.sort(key=version_key, reverse=True)

        for model_name in candidates:
            try:
                return genai.GenerativeModel(model_name)
            except Exception:
                continue

        raise Exception("No usable Gemini model found.")