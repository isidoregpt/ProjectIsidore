# This program is licensed under the GNU General Public License v3.0.
# For more details, see: https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author: Jonathan Graziola (isidore.gpt@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import streamlit as st
import datetime
import io
import zipfile
import os
import tempfile
from fpdf import FPDF
import pandas as pd
from openai import OpenAI
import requests
import json
import time

# Model provider configurations including parameter support
MODEL_OPTIONS = {
    "OpenAI": {
        "models": [
            "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14",
            "gpt-4.5-preview-2025-02-27", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18",
            "chatgpt-4o-latest", "o1-2024-12-17", "o3-mini-2025-01-31", "o4-mini-2025-04-16"
        ],
        "supports_temperature": True,
        "supports_top_p": True,
        "temp_range": (0.0, 2.0)
    },
    "Anthropic": {
        "models": ["claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"],
        "supports_temperature": True,
        "supports_top_p": False,
        "temp_range": (0.0, 1.0)
    }
}

class ModelManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.openai_client = OpenAI(api_key=api_key)
        self.vector_store_id = None
        self.vector_store_created = False

    def create_vector_store(self, prompt, rubric, reference):
        try:
            vs = self.openai_client.vector_stores.create(name="Essay Grading Context")
            self.vector_store_id = vs.id

            context_files = {"prompt.txt": prompt, "rubric.txt": rubric, "reference.txt": reference}
            for name, content in context_files.items():
                with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", encoding="utf-8", delete=False) as f:
                    f.write(content)
                    temp_filename = f.name
                with open(temp_filename, "rb") as f:
                    try:
                        self.openai_client.vector_stores.files.upload_and_poll(
                            vector_store_id=self.vector_store_id,
                            file=f
                        )
                    except Exception as upload_error:
                        st.warning(f"Error uploading {name}: {upload_error}")
                try:
                    os.remove(temp_filename)
                except:
                    pass

            self.vector_store_created = True
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            self.vector_store_created = False
            return False

    def get_vector_store_results(self, essay):
        try:
            if not self.vector_store_id:
                return None, False

            preview = essay[:100] if len(essay) > 100 else essay
            search_query = f"What criteria from the rubric apply to this essay? {preview}"

            results = self.openai_client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=search_query,
                max_num_results=5
            )

            formatted_results = []
            for result in results.data:
                content_text = "\n".join([part.text for part in result.content])
                formatted_results.append(f"From {result.filename}:\n{content_text}")

            rag_success = len(formatted_results) > 0
            return "\n\n".join(formatted_results), rag_success
        except Exception as e:
            st.warning(f"Vector store search had an issue: {e}")
            return None, False

    def grade_essay_with_openai(self, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        try:
            vector_store_context, rag_success = self.get_vector_store_results(essay)
            if vector_store_context:
                grading_request = (
                    f"Prompt:\n{prompt}\n\n"
                    f"Reference Material:\n{reference}\n\n"
                    f"Rubric:\n{rubric}\n\n"
                    f"Relevant Context:\n{vector_store_context}\n\n"
                    f"Essay to grade:\n{essay}\n\n"
                    "Grade this essay according to the rubric and reference material provided. "
                    "Be specific about points deducted and explain why. "
                    "First provide a detailed analysis, then summarize with total points deducted at the end."
                )
            else:
                grading_request = (
                    f"Prompt:\n{prompt}\n\n"
                    f"Reference Material:\n{reference}\n\n"
                    f"Rubric:\n{rubric}\n\n"
                    f"Essay to grade:\n{essay}\n\n"
                    "Grade this essay according to the rubric and reference material provided. "
                    "Be specific about points deducted and explain why. "
                    "First provide a detailed analysis, then summarize with total points deducted at the end."
                )

            request_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert essay grader who provides detailed feedback."},
                    {"role": "user", "content": grading_request}
                ]
            }

            if not any(tag in model for tag in ["o1", "o3-mini", "o4-mini"]):
                request_params["temperature"] = temperature
                request_params["top_p"] = top_p

            response = self.openai_client.chat.completions.create(**request_params)
            return {
                "content": response.choices[0].message.content,
                "vector_store_created": self.vector_store_created,
                "rag_success": rag_success
            }
        except Exception as e:
            return {"content": f"Error grading with OpenAI model {model}: {str(e)}", "vector_store_created": self.vector_store_created, "rag_success": False}

    def grade_essay_with_anthropic(self, model, prompt, rubric, reference, essay, temperature=0.7):
        try:
            vector_store_context, rag_success = self.get_vector_store_results(essay)
            if vector_store_context:
                grading_request = (
                    f"Prompt:\n{prompt}\n\n"
                    f"Reference Material:\n{reference}\n\n"
                    f"Rubric:\n{rubric}\n\n"
                    f"Relevant Context:\n{vector_store_context}\n\n"
                    f"Essay to grade:\n{essay}\n\n"
                    "Grade this essay according to the rubric and reference material provided. "
                    "Be specific about points deducted and explain why. "
                    "First provide a detailed analysis, then summarize with total points deducted at the end."
                )
            else:
                grading_request = (
                    f"Prompt:\n{prompt}\n\n"
                    f"Reference Material:\n{reference}\n\n"
                    f"Rubric:\n{rubric}\n\n"
                    f"Essay to grade:\n{essay}\n\n"
                    "Grade this essay according to the rubric and reference material provided. "
                    "Be specific about points deducted and explain why. "
                    "First provide a detailed analysis, then summarize with total points deducted at the end."
                )

            headers = {"Content-Type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
            data = {"model": model, "messages": [{"role": "user", "content": grading_request}], "temperature": temperature, "max_tokens": 4000}
            response = requests.post("https://api.anthropic.com/v1/messages", json=data, headers=headers)
            response.raise_for_status()
            return {"content": response.json()["content"][0]["text"], "vector_store_created": self.vector_store_created, "rag_success": rag_success}
        except Exception as e:
            return {"content": f"Error grading with Anthropic model {model}: {str(e)}", "vector_store_created": self.vector_store_created, "rag_success": False}

    def grade_essay(self, provider, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        if provider == "OpenAI":
            return self.grade_essay_with_openai(model, prompt, rubric, reference, essay, temperature, top_p)
        elif provider == "Anthropic":
            return self.grade_essay_with_anthropic(model, prompt, rubric, reference, essay, temperature)
        else:
            return {"content": f"Unsupported provider: {provider}", "vector_store_created": False, "rag_success": False}


def save_pdf(essay_name, model_outputs):
    try:
        temp_dir = tempfile.mkdtemp()
        temp_pdf_path = os.path.join(temp_dir, f"graded_{essay_name}.pdf")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        safe_name = ''.join(c if ord(c) < 128 else '?' for c in essay_name)
        pdf.cell(200, 10, f"Graded Essay: {safe_name}", ln=True, align="C")

        for model, output_data in model_outputs.items():
            pdf.set_font('Arial', 'B', 12)
            safe_model = ''.join(c if ord(c) < 128 else '?' for c in model)
            rag_status = (f"Vector Store: {'Success' if output_data['vector_store_created'] else 'Failed'}, "
                          f"RAG: {'Success' if output_data['rag_success'] else 'Failed'}")
            pdf.cell(200, 10, f"Model: {safe_model}", ln=True)
            pdf.cell(200, 10, f"RAG Status: {rag_status}", ln=True)
            pdf.set_font('Arial', '', 10)
            for line in output_data['content'].split("\n"):
                safe_line = ''.join(c if ord(c) < 128 else '?' for c in line)
                pdf.multi_cell(0, 5, safe_line)
            pdf.ln()

        pdf.output(temp_pdf_path)
        with open(temp_pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        try:
            os.remove(temp_pdf_path)
            os.rmdir(temp_dir)
        except:
            pass
        return pdf_bytes
    except Exception as e:
        st.warning(f"PDF generation failed: {e}. Using text format instead.")
        return None


def display_comparison_table(results):
    try:
        total_scores = []
        rows = []
        for model, output_data in results.items():
            content = output_data['content']
            vector_store_created = output_data['vector_store_created']
            rag_success = output_data['rag_success']
            lines = content.splitlines()
            deduction_line = next((line for line in lines if "points deducted" in line.lower()), "Not Found")
            summary = next((line for line in lines[::-1] if line.strip()), "")
            try:
                digits = ''.join(c for c in deduction_line if c.isdigit() or c == '.')
                points = float(digits) if digits else 0
            except:
                points = 0
            total_scores.append((model, points))
            rag_status = (f"Vector Store: {'Success' if vector_store_created else 'Failed'}, "
                          f"RAG: {'Success' if rag_success else 'Failed'}")
            rows.append({"Model": model, "RAG Status": rag_status, "Points Deducted": deduction_line, "Summary": summary[:150]})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        if total_scores:
            st.subheader("📊 Average Points Deducted")
            score_df = pd.DataFrame(total_scores, columns=["Model", "Points Deducted"])EOF
