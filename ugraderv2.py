# This program is licensed under the GNU General Public License v3.0.
# For more details, see: https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author: Jonathan Graziola (isidore.gpt@gmail.com)
# Modified by Gemini to include Google Gemini models and integrate fixes.
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
import google.generativeai as genai # Added for Gemini
import traceback # For detailed error logging

# Model provider configurations including parameter support
MODEL_OPTIONS = {
    "OpenAI": {
        "models": [
            # Example models, update as needed
             "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
            # "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14",
            # "gpt-4.5-preview-2025-02-27", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18",
            # "chatgpt-4o-latest", "o1-2024-12-17", "o3-mini-2025-01-31", "o4-mini-2025-04-16"
        ],
        "supports_temperature": True,
        "supports_top_p": True,
        "temp_range": (0.0, 2.0)
    },
    "Anthropic": {
        "models": [
            # Example models, update as needed
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"
            # "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"
            ],
        "supports_temperature": True,
        "supports_top_p": True, # Anthropic API generally supports top_p via messages API now
        "temp_range": (0.0, 1.0)
    },
    "Google": {
        "models": [
             # Example models, update as needed
             "gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro"
             ],
        "supports_temperature": True,
        "supports_top_p": True,
        "temp_range": (0.0, 1.0) # Standard range for Gemini (sometimes up to 2.0)
    }
}

class ModelManager:
    def __init__(self, api_key):
        self.api_key = api_key # Store the common API key
        # Initialize OpenAI client (required for Vector Store)
        try:
            self.openai_client = OpenAI(api_key=api_key)
            self.openai_client_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client (needed for Vector Store): {e}")
            self.openai_client = None
            self.openai_client_initialized = False

        self.vector_store_id = None
        self.vector_store_created = False
        # Note: Gemini client configured on-demand in its specific method

    def create_vector_store(self, prompt, rubric, reference):
        """Create vector store for retrieval-augmented grading (using OpenAI's service)"""
        if not self.openai_client_initialized:
             st.warning("OpenAI client not initialized. Skipping Vector Store creation.")
             self.vector_store_created = False
             return False
        try:
            # Create a new vector store
            vs = self.openai_client.vector_stores.create(
                name=f"Essay Grading Context - {datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
            self.vector_store_id = vs.id
            st.info(f"Attempting to create Vector Store (ID: {self.vector_store_id})...")

            context_files = {
                "prompt.txt": prompt,
                "rubric.txt": rubric,
                "reference.txt": reference
            }

            uploaded_file_ids = [] # Track names of successfully uploaded files
            temp_files_to_clean = []
            any_upload_failed = False
            files_processed_count = 0

            for name, content in context_files.items():
                if not content or content.isspace():
                    st.warning(f"Skipping empty file: {name}")
                    continue

                temp_filename = None # Initialize for cleanup safety
                try:
                    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", encoding="utf-8", delete=False) as f:
                        f.write(content)
                        temp_filename = f.name
                        temp_files_to_clean.append(temp_filename)
                        st.write(f"  - Created temporary file for {name}: {temp_filename}")
                except Exception as temp_file_error:
                    st.error(f"Error creating temporary file for {name}: {temp_file_error}")
                    any_upload_failed = True
                    continue # Skip this file

                # Reopen in binary mode for upload
                try:
                    with open(temp_filename, "rb") as f_bin:
                        st.write(f"  - Uploading {name} to Vector Store {self.vector_store_id}...")

                        # --- Using the file upload method from the "worked" code ---
                        # This uploads files individually without using .beta
                        # Note: This might be slightly less efficient than batching but avoids the .beta error
                        # And uses the specific method confirmed to work by the user.
                        file_object = self.openai_client.files.create(file=f_bin, purpose='assistants') # Changed purpose to assistants
                        # Now add the file to the vector store
                        vector_store_file = self.openai_client.vector_stores.files.create(
                            vector_store_id=self.vector_store_id,
                            file_id=file_object.id
                        )
                        # --- End file upload section ---

                        # Poll for file processing status within the store
                        # This is a bit more complex than upload_and_poll, requires checking file status
                        max_wait_time = 120 # seconds
                        start_time = time.time()
                        file_status = "in_progress"
                        while file_status != "completed" and time.time() - start_time < max_wait_time:
                             try:
                                 retrieved_file = self.openai_client.vector_stores.files.retrieve(
                                     vector_store_id=self.vector_store_id,
                                     file_id=vector_store_file.id
                                 )
                                 # Check for status attribute (might vary slightly based on API version)
                                 if hasattr(retrieved_file, 'status'):
                                     file_status = retrieved_file.status
                                 elif retrieved_file.last_error: # Check for errors
                                     file_status = "failed"
                                     st.warning(f"File processing error for {name}: {retrieved_file.last_error}")
                                     break
                                 else: # Assume in progress if status field missing and no error
                                      file_status = "in_progress"

                                 if file_status != "completed":
                                     time.sleep(5) # Wait before polling again

                             except Exception as poll_error:
                                 st.warning(f"Error polling status for {name}: {poll_error}. Assuming failure.")
                                 file_status = "failed"
                                 break

                        if file_status == "completed":
                            uploaded_file_ids.append(name)
                            st.write(f"    - Upload and processing successful for {name}.")
                            files_processed_count += 1
                        else:
                             st.warning(f"File processing for {name} did not complete successfully. Final status: {file_status}")
                             any_upload_failed = True
                             # Attempt to delete the failed file from the store if possible
                             try:
                                 self.openai_client.vector_stores.files.delete(vector_store_id=self.vector_store_id, file_id=vector_store_file.id)
                                 st.write(f"    - Attempted to remove failed file {name} from vector store.")
                             except Exception:
                                 st.write(f"    - Could not remove failed file {name} from vector store.")


                except Exception as upload_error:
                    st.error(f"Error uploading {name} (file ID: {file_object.id if 'file_object' in locals() else 'N/A'}) to vector store {self.vector_store_id}: {upload_error}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    any_upload_failed = True

            # Clean up temporary files
            for temp_file in temp_files_to_clean:
                 try:
                     os.remove(temp_file)
                 except Exception as cleanup_error:
                     st.warning(f"Could not remove temporary file {temp_file}: {cleanup_error}")

            # Check overall success
            if any_upload_failed or files_processed_count == 0:
                 st.error("One or more files failed to upload/process, or no files were added. Vector Store may be incomplete or unusable.")
                 try:
                     st.write(f"Attempting to delete incomplete vector store {self.vector_store_id}...")
                     delete_response = self.openai_client.vector_stores.delete(self.vector_store_id)
                     if delete_response.deleted:
                          st.info(f"Incomplete vector store {self.vector_store_id} deleted.")
                     else:
                          st.warning(f"Failed to confirm deletion of incomplete vector store {self.vector_store_id}.")
                 except Exception as delete_error:
                     st.warning(f"Could not delete incomplete vector store {self.vector_store_id}: {delete_error}")
                 self.vector_store_id = None
                 self.vector_store_created = False
                 return False

            self.vector_store_created = True
            st.success(f"Vector store created successfully (ID: {self.vector_store_id}) with {files_processed_count} file(s): {', '.join(uploaded_file_ids)}.")
            return True

        except Exception as e:
            st.error(f"Critical Error during vector store creation: {e}")
            st.error(f"Traceback: {traceback.format_exc()}")
            self.vector_store_created = False
            if self.vector_store_id: # Attempt cleanup if VS ID was obtained
                try:
                    st.write(f"Attempting cleanup of partially created vector store {self.vector_store_id}...")
                    self.openai_client.vector_stores.delete(self.vector_store_id)
                    st.info(f"Cleaned up partially created vector store {self.vector_store_id}.")
                except Exception as delete_error:
                    st.warning(f"Failed to clean up vector store {self.vector_store_id} after error: {delete_error}")
            self.vector_store_id = None
            return False

    def get_vector_store_results(self, essay):
        """Use Chat Completion as a proxy to query the vector store for relevant context."""
        if not self.openai_client_initialized or not self.vector_store_id:
            return None, False
        try:
            preview = essay[:300] if len(essay) > 300 else essay
            # Create an assistant specialized for this task (can be done once or reused)
            # For simplicity here, creating it each time, but could be optimized
            try:
                assistant = self.openai_client.beta.assistants.create(
                    name="Essay Context Retriever",
                    instructions="You are an assistant designed to find relevant sections from provided files (prompt, rubric, reference) in a vector store based on an essay excerpt. Focus on criteria directly applicable to grading.",
                    model="gpt-4o-mini", # Use a fast, cheaper model for retrieval
                    tool_resources={"file_search": {"vector_store_ids": [self.vector_store_id]}},
                    tools=[{"type": "file_search"}]
                )
                assistant_id = assistant.id
            except Exception as assistant_error:
                st.warning(f"Could not create/use dedicated assistant for retrieval: {assistant_error}. Falling back to direct chat query if possible.")
                # Fallback: Use simpler chat completion without explicit file search tool if assistant fails
                # This is less reliable for finding *specific file content*
                search_query = (
                     f"Based on the typical content of an essay prompt, grading rubric, and reference material for essay evaluation, "
                     f"identify 3-5 potentially relevant criteria or points for evaluating an essay starting with this excerpt: '{preview}...'."
                )
                retrieval_response = self.openai_client.chat.completions.create(
                     model="gpt-4o-mini",
                     messages=[
                          {"role": "system", "content": "You are an assistant helping to identify relevant grading criteria based on an essay excerpt."},
                          {"role": "user", "content": search_query}
                     ],
                     temperature=0.1, max_tokens=300
                )
                formatted_results = retrieval_response.choices[0].message.content
                rag_success = bool(formatted_results and formatted_results.strip())
                st.write("  - RAG context retrieval via fallback Chat Completion.")
                return formatted_results, rag_success


            # Create a thread
            thread = self.openai_client.beta.threads.create()

            # Add user message to the thread
            self.openai_client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Find relevant context from the attached files to help grade an essay starting with: '{preview}...'"
            )

            # Run the assistant
            run = self.openai_client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant_id,
                 max_completion_tokens=300 # Limit output tokens
            )

            formatted_results = ""
            rag_success = False
            if run.status == 'completed':
                messages = self.openai_client.beta.threads.messages.list(thread_id=thread.id, order='asc')
                # Extract the assistant's response, potentially including citations
                for msg in messages:
                     if msg.role == "assistant":
                         for content_block in msg.content:
                             if content_block.type == 'text':
                                 formatted_results += content_block.text.value
                                 # Check for citations if needed (complex to parse reliably)
                                 # annotations = content_block.text.annotations
                                 # if annotations: formatted_results += "\n[See citations]"
                         break # Usually only one assistant message per run
                rag_success = bool(formatted_results and formatted_results.strip())
                st.write("  - RAG context retrieval via Assistant API successful.")
            else:
                st.warning(f"Assistant run for RAG failed or did not complete. Status: {run.status}")

            # Clean up the temporary assistant (optional, depends on usage pattern)
            try:
                self.openai_client.beta.assistants.delete(assistant_id)
            except Exception:
                pass # Ignore cleanup error


            return formatted_results, rag_success

        except Exception as e:
            st.warning(f"Vector store context retrieval via Assistant API had an issue: {e}")
            st.warning(f"Traceback: {traceback.format_exc()}")
            return None, False


    def grade_essay_with_openai(self, model, prompt, rubric, reference, essay, temperature=0.1, top_p=0.9):
        """Grade an essay using an OpenAI model with retrieved context"""
        if not self.openai_client_initialized:
            return {
                "content": "Error: OpenAI client not initialized.",
                "vector_store_created": self.vector_store_created,
                "rag_success": False
            }
        try:
            # Try to get relevant context from vector store (via Assistant method)
            vector_store_context, rag_success = self.get_vector_store_results(essay)

            # Prepare the grading request
            grading_request_parts = [
                f"**Prompt:**\n{prompt}\n",
                f"**Reference Material:**\n{reference}\n",
                f"**Rubric:**\n{rubric}\n"
            ]
            if rag_success and vector_store_context:
                grading_request_parts.append(f"**Relevant Context Retrieved (Use this heavily for grading):**\n{vector_store_context}\n")
            else:
                 grading_request_parts.append("**Relevant Context Retrieved:** None (or retrieval failed) - Grade based on prompt/rubric/reference only.\n")

            grading_request_parts.extend([
                 f"\n**Essay to grade:**\n{essay}\n\n",
                 "--- **TASK** ---",
                 "You are an expert essay grader focusing on detailed, rubric-based feedback.",
                 "1. Grade this essay **strictly** according to the Rubric, using the Prompt and Reference Material.",
                 "2. **Prioritize the 'Relevant Context Retrieved' section** if provided; it contains the most applicable criteria found for this specific essay.",
                 "3. Be specific about points deducted and explain *why*, referencing the rubric criteria directly.",
                 "4. Provide a detailed analysis first.",
                 "5. Then, at the very end, on a new line, summarize with ONLY the format 'Total Points Deducted: X' where X is a number (e.g., 'Total Points Deducted: 5' or 'Total Points Deducted: 2.5')."
             ])
            grading_request = "".join(grading_request_parts)

            request_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert essay grader focusing on detailed, rubric-based feedback, adhering strictly to the provided rubric, reference, and retrieved context (if available)."},
                    {"role": "user", "content": grading_request}
                ],
                 "max_tokens": 4000
            }

            provider_config = MODEL_OPTIONS.get("OpenAI", {})
            if provider_config.get("supports_temperature", False):
                 min_temp, max_temp = provider_config.get("temp_range", (0.0, 1.0))
                 request_params["temperature"] = max(min_temp, min(temperature, max_temp))
            if provider_config.get("supports_top_p", False):
                 request_params["top_p"] = top_p

            response = self.openai_client.chat.completions.create(**request_params)

            return {
                "content": response.choices[0].message.content,
                "vector_store_created": self.vector_store_created,
                "rag_success": rag_success
            }

        except Exception as e:
            st.error(f"Error grading with OpenAI model {model}: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
            return {
                "content": f"Error grading with OpenAI model {model}: {str(e)}",
                "vector_store_created": self.vector_store_created,
                "rag_success": False
            }

    def grade_essay_with_anthropic(self, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        """Grade an essay using an Anthropic model, reusing OpenAI RAG context"""
        try:
            # Reuse context retrieved via OpenAI Assistant method
            vector_store_context, rag_success = self.get_vector_store_results(essay)

            # Prepare the grading request (similar structure)
            system_prompt_anthropic = "You are an expert essay grader focusing on detailed, rubric-based feedback, adhering strictly to the provided rubric, reference, and retrieved context (if available)."
            user_prompt_parts = [
                 f"**Prompt:**\n{prompt}\n",
                 f"**Reference Material:**\n{reference}\n",
                 f"**Rubric:**\n{rubric}\n"
             ]
            if rag_success and vector_store_context:
                user_prompt_parts.append(f"**Relevant Context Retrieved (Use this heavily for grading):**\n{vector_store_context}\n")
            else:
                user_prompt_parts.append("**Relevant Context Retrieved:** None (or retrieval failed) - Grade based on prompt/rubric/reference only.\n")

            user_prompt_parts.extend([
                 f"\n**Essay to grade:**\n{essay}\n\n",
                 "--- **TASK** ---",
                 "1. Grade this essay **strictly** according to the Rubric, using the Prompt and Reference Material.",
                 "2. **Prioritize the 'Relevant Context Retrieved' section** if provided; it contains the most applicable criteria found for this specific essay.",
                 "3. Be specific about points deducted and explain *why*, referencing the rubric criteria directly.",
                 "4. Provide a detailed analysis first.",
                 "5. Then, at the very end, on a new line, summarize with ONLY the format 'Total Points Deducted: X' where X is a number (e.g., 'Total Points Deducted: 5' or 'Total Points Deducted: 2.5')."
             ])
            user_prompt_anthropic = "".join(user_prompt_parts)

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01" # Consider updating if newer versions offer benefits
            }

            data = {
                "model": model,
                "system": system_prompt_anthropic,
                "messages": [{"role": "user", "content": user_prompt_anthropic}],
                "max_tokens": 4000
            }

            provider_config = MODEL_OPTIONS.get("Anthropic", {})
            if provider_config.get("supports_temperature", False):
                 min_temp, max_temp = provider_config.get("temp_range", (0.0, 1.0))
                 data["temperature"] = max(min_temp, min(temperature, max_temp))
            if provider_config.get("supports_top_p", False):
                 if top_p is not None:
                      data["top_p"] = top_p

            response = requests.post("https://api.anthropic.com/v1/messages", json=data, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            if response_data.get("type") == "error":
                error_details = response_data.get("error", {})
                raise ValueError(f"Anthropic API Error: Type: {error_details.get('type', 'N/A')}, Message: {error_details.get('message', 'No message provided')}")
            if "content" not in response_data or not isinstance(response_data["content"], list) or len(response_data["content"]) == 0:
                 raise ValueError("Unexpected response format from Anthropic API: 'content' missing or invalid.")
            if response_data["content"][0].get("type") == "text":
                content = response_data["content"][0]["text"]
            else:
                raise ValueError("Expected text content from Anthropic API, but received different type.")

            return {
                "content": content,
                "vector_store_created": self.vector_store_created,
                "rag_success": rag_success
            }

        except requests.exceptions.RequestException as e:
             error_content = f"API Request Error: {str(e)}"
             if e.response is not None:
                  error_content += f"\nStatus Code: {e.response.status_code}"
                  try: error_content += f"\nDetails: {json.dumps(e.response.json())}"
                  except json.JSONDecodeError: error_content += f"\nResponse Body: {e.response.text}"
             st.error(f"Error grading with Anthropic model {model}: {error_content}")
             # st.error(f"Traceback: {traceback.format_exc()}") # Maybe too verbose for UI
             return { "content": f"Error: {error_content}", "vector_store_created": self.vector_store_created, "rag_success": False }
        except Exception as e:
             st.error(f"Error grading with Anthropic model {model}: {str(e)}")
             # st.error(f"Traceback: {traceback.format_exc()}")
             return { "content": f"Error: {str(e)}", "vector_store_created": self.vector_store_created, "rag_success": False }


    def grade_essay_with_google(self, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        """Grade an essay using a Google Gemini model, reusing OpenAI RAG context"""
        try:
            try:
                 genai.configure(api_key=self.api_key)
            except Exception as config_error:
                 st.error(f"Error configuring Google Gemini client: {str(config_error)}")
                 return { "content": f"Error configuring Gemini: {str(config_error)}", "vector_store_created": self.vector_store_created, "rag_success": False }

            # Reuse context retrieved via OpenAI Assistant method
            vector_store_context, rag_success = self.get_vector_store_results(essay)

            system_instruction = (
                "You are an expert essay grader focusing on detailed, rubric-based feedback. "
                "Adhere strictly to the provided rubric, reference material, and retrieved context (if available)."
            )
            user_prompt_parts = [
                 f"**Prompt:**\n{prompt}\n",
                 f"**Reference Material:**\n{reference}\n",
                 f"**Rubric:**\n{rubric}\n"
             ]
            if rag_success and vector_store_context:
                user_prompt_parts.append(f"**Relevant Context Retrieved (Use this heavily for grading):**\n{vector_store_context}\n")
            else:
                user_prompt_parts.append("**Relevant Context Retrieved:** None (or retrieval failed) - Grade based on prompt/rubric/reference only.\n")

            user_prompt_parts.extend([
                 f"\n**Essay to grade:**\n{essay}\n\n",
                 "--- **TASK** ---",
                 "1. Grade this essay **strictly** according to the Rubric, using the Prompt and Reference Material.",
                 "2. **Prioritize the 'Relevant Context Retrieved' section** if provided; it contains the most applicable criteria found for this specific essay.",
                 "3. Be specific about points deducted and explain *why*, referencing the rubric criteria directly.",
                 "4. Provide a detailed analysis first.",
                 "5. Then, at the very end, on a new line, summarize with ONLY the format 'Total Points Deducted: X' where X is a number (e.g., 'Total Points Deducted: 5' or 'Total Points Deducted: 2.5')."
             ])
            full_user_prompt = "".join(user_prompt_parts)

            try:
                gemini_model = genai.GenerativeModel(model, system_instruction=system_instruction)
            except TypeError:
                gemini_model = genai.GenerativeModel(model)
                st.warning("Gemini model/SDK might not support 'system_instruction' in constructor.")

            provider_config = MODEL_OPTIONS.get("Google", {})
            gen_config_params = {}
            if provider_config.get("supports_temperature", False):
                 min_temp, max_temp = provider_config.get("temp_range", (0.0, 1.0))
                 effective_max_temp = min(max_temp, 1.0)
                 gen_config_params["temperature"] = max(min_temp, min(temperature, effective_max_temp))
            if provider_config.get("supports_top_p", False):
                 if top_p is not None: gen_config_params["top_p"] = top_p

            generation_config = genai.types.GenerationConfig(**gen_config_params)
            safety_settings = [
                 {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             ]

            response = gemini_model.generate_content(
                 full_user_prompt,
                 generation_config=generation_config,
                 safety_settings=safety_settings
            )

            try:
                 content = response.text
            except ValueError as e:
                 block_reason, safety_feedback_str, finish_reason = "Unknown", "N/A", "Unknown"
                 try:
                     if response.prompt_feedback:
                          block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback.block_reason else "Not Specified"
                          safety_feedback_str = str(response.prompt_feedback.safety_ratings)
                 except (AttributeError, ValueError): pass
                 try:
                      if response.candidates and response.candidates[0].finish_reason: finish_reason = response.candidates[0].finish_reason.name
                 except (AttributeError, IndexError, ValueError): pass
                 content = (f"Content generation issue. Finish Reason: {finish_reason}.\n"
                            f"Safety Block Reason: {block_reason}.\nSafety Ratings: {safety_feedback_str}\nError hint: {str(e)}")
                 rag_success = False
                 st.warning(f"Gemini content issue for {model}. Finish: {finish_reason}, Block: {block_reason}")
            except Exception as e:
                 content = f"Error accessing Gemini response content: {str(e)}"
                 rag_success = False
                 st.error(f"Error accessing Gemini response text for {model}: {e}")

            return {
                "content": content,
                "vector_store_created": self.vector_store_created,
                "rag_success": rag_success
            }

        except Exception as e:
            st.error(f"Error grading with Google model {model}: {str(e)}")
            # st.error(f"Traceback: {traceback.format_exc()}")
            return { "content": f"Error: {str(e)}", "vector_store_created": self.vector_store_created, "rag_success": False }

    def grade_essay(self, provider, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        """Route the grading request to the appropriate API based on provider"""
        if provider == "OpenAI":
            return self.grade_essay_with_openai(model, prompt, rubric, reference, essay, temperature, top_p)
        elif provider == "Anthropic":
            return self.grade_essay_with_anthropic(model, prompt, rubric, reference, essay, temperature, top_p)
        elif provider == "Google":
             return self.grade_essay_with_google(model, prompt, rubric, reference, essay, temperature, top_p)
        else:
            st.error(f"Unsupported provider: {provider}")
            return { "content": f"Unsupported provider: {provider}", "vector_store_created": False, "rag_success": False }


def save_pdf(essay_name, model_outputs):
    """Generate a PDF with grading results and RAG status using standard fonts"""
    try:
        temp_dir = tempfile.mkdtemp()
        safe_essay_name_for_file = ''.join(c for c in essay_name if c.isalnum() or c in ('_', '-')).strip()[:100]
        if not safe_essay_name_for_file: safe_essay_name_for_file = "graded_essay"
        temp_pdf_path = os.path.join(temp_dir, f"{safe_essay_name_for_file}.pdf")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        safe_title_display = essay_name.encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(0, 10, f"Graded Essay: {safe_title_display}", ln=True, align="C")
        pdf.ln(5)

        for model, output_data in model_outputs.items():
            pdf.set_font('Arial', 'B', 12)
            safe_model_display = model.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 10, f"--- Model: {safe_model_display} ---", ln=True)

            rag_status = (f"Vector Store: {'Created' if output_data.get('vector_store_created', False) else 'Not Created/Failed'} | "
                           f"RAG Context: {'Retrieved' if output_data.get('rag_success', False) else 'Not Retrieved/Failed/Skipped'}") # Updated text
            pdf.set_font('Arial', 'I', 9)
            safe_rag_status = rag_status.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 8, f"Status: {safe_rag_status}", ln=True)
            pdf.ln(2)

            pdf.set_font('Arial', '', 10)
            content = output_data.get("content", "Error: No content found.")
            for line in content.split("\n"):
                 safe_line = line.encode('latin-1', 'replace').decode('latin-1')
                 pdf.multi_cell(0, 5, safe_line)
            pdf.ln(5)

        pdf.output(temp_pdf_path)
        with open(temp_pdf_path, 'rb') as f: pdf_bytes = f.read()
        try:
            os.remove(temp_pdf_path)
            os.rmdir(temp_dir)
        except Exception as cleanup_error: st.warning(f"Could not clean up temp PDF: {cleanup_error}")
        return pdf_bytes
    except Exception as e:
        st.warning(f"PDF generation failed for '{essay_name}': {e}. Results in TXT/CSV only.")
        # st.warning(f"Traceback: {traceback.format_exc()}") # Optional: for debugging PDF issues
        return None


def display_comparison_table(results):
    """Display results table more robustly and return dataframe for export"""
    try:
        total_scores = []
        rows = []
        for model, output_data in results.items():
            content = output_data.get("content", "Error: No content.")
            if not isinstance(content, str): content = str(content)
            vector_store_created = output_data.get("vector_store_created", False)
            rag_success = output_data.get("rag_success", False)

            lines = content.splitlines()
            deduction_line_text = "Not Found"
            points = None
            for line in reversed(lines):
                 line_lower = line.lower().strip()
                 if line_lower.startswith("total points deducted:"):
                      deduction_line_text = line.strip()
                      try:
                           numeric_part_str = line.split(":")[-1].strip()
                           cleaned_numeric_str = ''.join(c for c in numeric_part_str if c.isdigit() or c == '.' or (c == '-' and numeric_part_str.startswith('-')))
                           if cleaned_numeric_str and cleaned_numeric_str != '-' and cleaned_numeric_str != '.':
                                points = float(cleaned_numeric_str)
                      except: points = None
                      break # Found line

            summary = ""
            non_empty_lines = [l.strip() for l in lines if l.strip() and not l.lower().strip().startswith("total points deducted:")]
            if non_empty_lines: summary = "\n".join(non_empty_lines[:3]) + ("..." if len(non_empty_lines) > 3 else "")

            if points is not None: total_scores.append((model, points))

            rag_status_str = (f"Store: {'Created' if vector_store_created else 'Not Created/Failed'} | "
                               f"RAG: {'Retrieved' if rag_success else 'Not Retrieved/Failed/Skipped'}") # Updated text

            rows.append({
                "Model": model, "RAG Status": rag_status_str,
                "Points Deducted Info": deduction_line_text,
                "Extracted Points": f"{points:.1f}" if points is not None else "N/A",
                "Feedback Summary": summary[:250] + ('...' if len(summary)>250 else '')
            })

        if not rows:
            st.warning("No results to display.")
            return None

        df = pd.DataFrame(rows)[["Model", "RAG Status", "Extracted Points", "Points Deducted Info", "Feedback Summary"]]
        st.dataframe(df, use_container_width=True)

        if total_scores:
            st.subheader("📊 Average Points Deducted (Parsed Scores)")
            score_df = pd.DataFrame(total_scores, columns=["Model", "Points Deducted"])
            avg_std = score_df.groupby("Model")["Points Deducted"].agg(['mean', 'std', 'count']).reset_index()
            avg_std.rename(columns={'mean': 'Avg Deduct', 'std': 'Std Dev', 'count': 'Count'}, inplace=True)
            avg_std['Avg Deduct'] = avg_std['Avg Deduct'].map('{:.2f}'.format)
            avg_std['Std Dev'] = avg_std['Std Dev'].map('{:.2f}'.format)
            st.table(avg_std)

        return df
    except Exception as e:
        st.error(f"Error displaying comparison table: {e}")
        # st.error(f"Traceback: {traceback.format_exc()}")
        return None


def main():
    st.set_page_config(page_title="Multi-Model Essay Grader", layout="wide")
    st.title("🤖 Multi-Model Essay Grader with Vector Search")
    st.caption(f"Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Instructions Expander (using content from the refined version)
    with st.expander("💡 Instructions & Setup"):
         st.markdown("""
**Goal:** Grade essays using multiple AI models (OpenAI, Anthropic, Google Gemini) and compare outputs, potentially enhancing consistency with Retrieval-Augmented Generation (RAG) using OpenAI's Vector Store.

**Using the App Interface (Step-by-Step):**

1.  **Upload API Key:** Use **`🔑 API Key (TXT)`** to upload your API key file (used for *all* selected providers).
2.  **Upload Context:** Upload **`📝 Prompt (TXT)`**, **`📋 Rubric (TXT)`**, and **`📘 Reference (TXT)`**. *(Ensure UTF-8 encoding).*
3.  **Upload Essays:** Use **`📦 Upload ZIP of Essays`** or **`📂 Or Upload one or more Essay Files`**. *(Anonymize essays, ensure UTF-8).* Check info message for loaded count.
4.  **Select Models:** Expand provider sections under **`🤖 Choose Models for Grading`** and check desired models.
5.  **Adjust Settings (Optional):** Modify **`Temperature`** and **`Top-p Sampling`** under **`⚙️ Model Behavior Settings`**.
6.  **Start Grading:** Click **`🚀 Grade Essays`**.
7.  **Wait & Monitor:** Observe progress bars and status messages (Vector Store creation, RAG retrieval per essay).
8.  **Review Results:** For each essay: check **`RAG Status`**, **`Comparison Table`**, and **`Average Points Deducted`**.
9.  **Download Outputs:** Use individual **`📥 Download Results...`** buttons or the final **`📦 Download ALL Results (Single ZIP)`** button.

**Understanding the Process:**

* An **OpenAI Vector Store** is created once using Prompt, Rubric, Reference.
* For each essay, **RAG Context** is retrieved from the store (using OpenAI Assistant API).
* Essay, original context, AND retrieved RAG context are sent to each selected model (OpenAI, Anthropic, Google).
* **RAG Status** table shows store creation success (overall) and RAG retrieval success (per essay).
        """)

    # About Expander (using content from the refined version)
    with st.expander("📚 About & Disclaimer"):
        st.markdown("""
**Multi-Model Essay Grader with Retrieval-Augmented Generation**

Experiment with AI-assisted grading using OpenAI, Anthropic, and Google Gemini models. Leverages OpenAI's vector stores and Assistant API for retrieval-augmented generation (RAG) to improve grading consistency.

**Key Features:**
- Multi-Provider Support: OpenAI, Anthropic, Google Gemini.
- Vector Store RAG (via OpenAI): Creates vector store from context, uses Assistant API for RAG retrieval per essay.
- Context Injection: Retrieved context added to prompts for all selected models.
- Comprehensive Output: Feedback, points deducted, RAG status.
- Consolidated Results: Download TXT, PDF, CSV per essay or as a single ZIP.

**How Vector Storage & RAG Work Here:**
1. OpenAI Vector Store created once from uploaded context.
2. Assistant API queries store for relevant context per essay.
3. Retrieved context included in prompts sent to all selected models.
4. 'RAG Status' table shows store success (overall) & retrieval success (per essay).

**Data Privacy:**
- Files processed in memory; no server-side storage by this app.
- Vector stores temporary via OpenAI. App attempts cleanup on failure.
- Data sent to selected API providers (OpenAI, Anthropic, Google). Review their policies.
- **Anonymize student essays before uploading.**

**Important Notes:**
- **API Key:** One key used for all providers. Ensure compatibility.
- **Evaluation Tool:** Verify AI output; not a replacement for human judgment.
- **Compliance:** Anonymize data per privacy laws (FERPA, GDPR).
- **PDF Output:** Basic fonts; special characters may be replaced.

**License & Author:**
Jonathan Graziola (isidore.gpt@gmail.com), modified by Gemini. GNU GPL v3.0.

**Disclaimer:** Use responsibly. NO WARRANTY provided.
        """)

    col1, col2, col3, col4 = st.columns(4)
    with col1: api_key_file = st.file_uploader("🔑 API Key (TXT)", type=["txt"])
    with col2: prompt_file = st.file_uploader("📝 Prompt (TXT)", type=["txt"])
    with col3: rubric_file = st.file_uploader("📋 Rubric (TXT)", type=["txt"])
    with col4: reference_file = st.file_uploader("📘 Reference (TXT)", type=["txt"]) # Changed label slightly

    st.markdown("---")
    st.subheader("📂 Upload Essays")
    essay_zip = st.file_uploader("📦 Upload ZIP of Essays (TXT files, UTF-8)", type=["zip"])

    essay_files = []
    essay_names = []
    # Use the more robust file handling from previous version
    if essay_zip:
        try:
            with zipfile.ZipFile(essay_zip) as z:
                all_files = z.namelist()
                essay_names_in_zip = [f for f in all_files if f.endswith(".txt") and not f.startswith('__MACOSX/') and not os.path.basename(f).startswith('.') and '/' not in f]
                if not essay_names_in_zip: st.warning("No '.txt' files found in ZIP root.")
                temp_essay_files, temp_essay_names = [], []
                for f_name in essay_names_in_zip:
                    try:
                        file_content = z.read(f_name).decode("utf-8", errors="replace")
                        if file_content.strip():
                            temp_essay_files.append(file_content)
                            temp_essay_names.append(os.path.basename(f_name))
                        else: st.warning(f"Skipping empty file '{f_name}' from ZIP.")
                    except Exception as decode_error: st.error(f"Error decoding '{f_name}': {decode_error}. Skipping.")
                essay_files, essay_names = temp_essay_files, temp_essay_names
        except zipfile.BadZipFile: st.error("Invalid ZIP file."); essay_zip = None
        except Exception as e: st.error(f"Error reading ZIP: {e}"); essay_zip = None
    else:
        uploaded_files = st.file_uploader("📂 Or Upload one or more Essay Files (TXT, UTF-8)", type=["txt"], accept_multiple_files=True)
        if uploaded_files:
            temp_essay_files, temp_essay_names = [], []
            for f in uploaded_files:
                try:
                    file_content = f.read().decode("utf-8", errors="replace")
                    if file_content.strip():
                        temp_essay_files.append(file_content)
                        temp_essay_names.append(f.name)
                    else: st.warning(f"Skipping empty file: {f.name}")
                except Exception as e: st.error(f"Error reading {f.name}: {e}. Skipping.")
            essay_files, essay_names = temp_essay_files, temp_essay_names

    if essay_files: st.info(f"Loaded {len(essay_files)} non-empty essay(s): {', '.join(essay_names)}")
    else: st.info("No essays loaded yet.")

    st.markdown("---")
    st.subheader("🤖 Choose Models for Grading")
    selected_models = []
    provider_columns = st.columns(len(MODEL_OPTIONS))
    col_index = 0
    for provider, provider_config in MODEL_OPTIONS.items():
        with provider_columns[col_index]:
             st.markdown(f"**{provider} Models**")
             for model in provider_config["models"]:
                 is_preview = "preview" in model.lower() or "latest" in model.lower()
                 label = f"{model}{' (Preview/Latest)' if is_preview else ''}"
                 if st.checkbox(label, key=f"model_{provider}_{model}"):
                     selected_models.append({"provider": provider, "model": model})
        col_index += 1

    if selected_models: st.write("Selected Models:", [f"{m['provider']} - {m['model']}" for m in selected_models])
    else: st.warning("No models selected.")

    st.markdown("---")
    st.subheader("⚙️ Model Behavior Settings")
    st.info("Settings applied only where supported by model/provider.")
    col_temp, col_top_p = st.columns(2)
    with col_temp: temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, help="0.0=deterministic, 1.0=creative")
    with col_top_p: top_p = st.slider("Top-p Sampling", 0.1, 1.0, 0.9, 0.05, help="Nucleus sampling threshold")

    st.markdown("---")

    if st.button("🚀 Grade Essays", type="primary"):
        # --- Input Validation ---
        valid_inputs = True
        if not api_key_file: st.error("❌ API Key required."); valid_inputs = False
        if not prompt_file: st.error("❌ Prompt file required."); valid_inputs = False
        if not rubric_file: st.error("❌ Rubric file required."); valid_inputs = False
        if not reference_file: st.error("❌ Reference file required."); valid_inputs = False
        if not essay_files: st.error("❌ At least one non-empty Essay required."); valid_inputs = False
        if not selected_models: st.error("❌ At least one Model required."); valid_inputs = False

        if valid_inputs:
            # --- Process Inputs ---
            try:
                api_key = api_key_file.read().decode("utf-8", errors="replace").strip()
                if not api_key: raise ValueError("API key file empty.")
                prompt = prompt_file.read().decode("utf-8", errors="replace")
                rubric = rubric_file.read().decode("utf-8", errors="replace")
                reference = reference_file.read().decode("utf-8", errors="replace")

                model_manager = ModelManager(api_key)

                # --- Create Vector Store (Once) ---
                vector_store_created_overall = False
                if model_manager.openai_client_initialized:
                     with st.spinner("Attempting OpenAI Vector Store creation..."):
                         vector_store_created_overall = model_manager.create_vector_store(prompt, rubric, reference)
                         if not vector_store_created_overall: st.warning("⚠️ Failed to create Vector Store. RAG skipped.")
                 else:
                     st.warning("⚠️ OpenAI client failed. Skipping Vector Store & RAG.")

                # --- Grade Each Essay ---
                all_essay_data_for_zip = []
                st.markdown("---"); st.header("📊 Grading Results")
                total_essays = len(essay_files)
                overall_progress_bar = st.progress(0, text="Starting...")

                for i, essay_text in enumerate(essay_files):
                    essay_name = essay_names[i] if i < len(essay_names) else f"essay_{i+1}.txt"
                    st.subheader(f"Processing Essay {i+1}/{total_essays}: {essay_name}")
                    overall_progress_bar.progress(i / total_essays, text=f"Processing {essay_name}...")

                    if not isinstance(essay_text, str) or not essay_text.strip():
                         st.warning(f"Skipping '{essay_name}' (empty/invalid).")
                         continue

                    results_this_essay = {}
                    model_progress_bar = st.progress(0, text=f"Starting models for '{essay_name}'...")
                    rag_status_container = st.container()
                    rag_statuses_this_essay = []

                    model_count = len(selected_models)
                    for idx, model_info in enumerate(selected_models):
                        provider = model_info["provider"]; model = model_info["model"]
                        model_key = f"{provider} - {model}"
                        progress_text = f"Grading '{essay_name}' with {model_key} ({idx+1}/{model_count})..."
                        model_progress_bar.progress((idx + 1) / model_count, text=progress_text)

                        try:
                            result_data = model_manager.grade_essay(
                                provider, model, prompt, rubric, reference, essay_text,
                                temperature=temperature, top_p=top_p
                            )
                            results_this_essay[model_key] = {
                                "content": result_data.get("content", "Error: Missing content"),
                                "vector_store_created": result_data.get("vector_store_created", vector_store_created_overall),
                                "rag_success": result_data.get("rag_success", False)
                            }
                            rag_statuses_this_essay.append({
                                 "Model": model_key,
                                 "Vector Store Status (Overall)": "Created" if results_this_essay[model_key]['vector_store_created'] else "Not Created/Failed",
                                 "RAG Context Retrieval (This Essay)": "Retrieved" if results_this_essay[model_key]['rag_success'] else "Not Retrieved/Failed/Skipped"
                            })
                        except Exception as grade_error:
                             st.error(f"Critical error grading {model_key}: {grade_error}")
                             results_this_essay[model_key] = { "content": f"Error: {str(grade_error)}", "vector_store_created": vector_store_created_overall, "rag_success": False }
                             rag_statuses_this_essay.append({ "Model": model_key, "Vector Store Status (Overall)": "Created" if vector_store_created_overall else "Not Created/Failed", "RAG Context Retrieval (This Essay)": "Failed (Error)" })
                        time.sleep(1.5) # Delay

                    with rag_status_container:
                         st.markdown("##### RAG Status (per model for this essay):")
                         if rag_statuses_this_essay: st.dataframe(pd.DataFrame(rag_statuses_this_essay), use_container_width=True)
                         else: st.write("No models processed.")
                    model_progress_bar.empty()
                    st.success(f"Finished models for '{essay_name}'.")

                    st.markdown("##### Comparison Table:")
                    results_df_this_essay = display_comparison_table(results_this_essay)

                    # --- Prepare & Download Files (This Essay) ---
                    essay_files_for_zip = {}
                    base_filename = ''.join(c for c in essay_name if c.isalnum() or c in ('_', '-')).strip()[:100]
                    if not base_filename: base_filename = f"essay_{i}"
                    # TXT
                    try:
                        text_content = f"ESSAY: {essay_name}\n\n{essay_text}\n\nGRADING:\n{'='*10}\n\n"
                        for m, d in results_this_essay.items():
                            text_content += f"== {m} ==\nStore: {'OK' if d.get('vector_store_created') else 'Fail'} | RAG: {'OK' if d.get('rag_success') else 'Fail/Skip'}\n---\n{d.get('content', 'Error')}\n===\n\n"
                        essay_files_for_zip["txt"] = (f"{base_filename}_graded.txt", text_content.encode('utf-8'))
                    except Exception as e: st.error(f"Error generating TXT for {essay_name}: {e}")
                    # PDF
                    pdf_bytes = save_pdf(essay_name, results_this_essay)
                    if pdf_bytes: essay_files_for_zip["pdf"] = (f"{base_filename}_graded.pdf", pdf_bytes)
                    # CSV
                    if results_df_this_essay is not None:
                         try:
                              csv_buffer = io.StringIO()
                              results_df_this_essay.to_csv(csv_buffer, index=False, encoding='utf-8')
                              essay_files_for_zip["csv"] = (f"{base_filename}_summary.csv", csv_buffer.getvalue().encode('utf-8'))
                         except Exception as e: st.error(f"Error generating CSV for {essay_name}: {e}")

                    all_essay_data_for_zip.append(essay_files_for_zip)

                    if essay_files_for_zip:
                         try:
                              essay_zip_buffer = io.BytesIO()
                              with zipfile.ZipFile(essay_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                                   for file_type, (filename, file_data) in essay_files_for_zip.items():
                                       if isinstance(file_data, bytes): zipf.writestr(filename, file_data)
                              essay_zip_buffer.seek(0)
                              st.download_button(f"📥 Download Results for '{essay_name}' (ZIP)", essay_zip_buffer, f"{base_filename}_results.zip", "application/zip", key=f"zip_{base_filename}_{i}")
                         except Exception as e: st.error(f"Error creating ZIP for {essay_name}: {e}")
                    st.markdown("---") # Separator

                overall_progress_bar.progress(1.0, text="Finished all essays!")

                # --- Final Download Button ---
                if all_essay_data_for_zip:
                     st.header(" Zipped Results (All Processed Essays)")
                     try:
                         all_essays_zip_buffer = io.BytesIO()
                         with zipfile.ZipFile(all_essays_zip_buffer, "w", zipfile.ZIP_DEFLATED) as bulk_zipf:
                              for essay_files_data in all_essay_data_for_zip:
                                  if isinstance(essay_files_data, dict):
                                       for file_type, (filename, file_data) in essay_files_data.items():
                                           if isinstance(file_data, bytes): bulk_zipf.writestr(filename, file_data)
                         all_essays_zip_buffer.seek(0)
                         st.download_button("📦 Download ALL Results (Single ZIP)", all_essays_zip_buffer, "ALL_graded_results.zip", "application/zip", key="zip_all")
                     except Exception as e: st.error(f"❌ Error creating combined ZIP: {e}")

            except ValueError as ve: st.error(f"❌ Input Error: {ve}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
