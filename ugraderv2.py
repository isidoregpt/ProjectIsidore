# This program is licensed under the GNU General Public License v3.0.
# For more details, see: https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author: Jonathan Graziola (isidore.gpt@gmail.com)
# Modified by Gemini to include Google Gemini models.
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
            # List of OpenAI models supported (Example models, update as needed)
             "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo" # Simplified common models
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
            # List of Anthropic models supported (Example models, update as needed)
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"
            # "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"
            ],
        "supports_temperature": True,
        "supports_top_p": True, # Anthropic API generally supports top_p via messages API now
        "temp_range": (0.0, 1.0)
    },
    # --- Updated Google Gemini Models (Reflecting latest additions) ---
    "Google": {
        "models": [
             "gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro" # Simplified common models
            # "gemini-2.5-pro-preview-03-25", # Added - Latest Pro Preview
            # "gemini-2.5-flash-preview-04-17",# Added - Latest Flash Preview
            # "gemini-2.0-flash",              # Added - Latest GA Flash
            # "gemini-1.5-pro",                # Kept - Stable Pro
            # "gemini-1.5-flash"               # Kept - Stable Flash
             ],
        "supports_temperature": True,
        "supports_top_p": True,
        "temp_range": (0.0, 1.0) # Standard range for Gemini (sometimes up to 2.0 depending on model/API version)
    }
    # --- End Google Gemini Update ---
    # Other providers can be added here as needed
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
                name=f"Essay Grading Context - {datetime.datetime.now().strftime('%Y%m%d%H%M%S')}" # Add timestamp for uniqueness
            )
            self.vector_store_id = vs.id
            st.info(f"Attempting to create Vector Store (ID: {self.vector_store_id})...")

            # Save context files temporarily
            context_files = {
                "prompt.txt": prompt,
                "rubric.txt": rubric,
                "reference.txt": reference
            }

            # Use tempfile to safely handle encoding issues
            uploaded_file_ids = [] # Track names of successfully uploaded files
            temp_files_to_clean = []
            any_upload_failed = False
            for name, content in context_files.items():
                # Check if content is empty or whitespace only
                if not content or content.isspace():
                    st.warning(f"Skipping empty file: {name}")
                    continue

                try:
                    # Create temp file with utf-8 encoding
                    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", encoding="utf-8", delete=False) as f:
                        f.write(content)
                        temp_filename = f.name
                        temp_files_to_clean.append(temp_filename) # Track for cleanup
                        st.write(f"  - Created temporary file for {name}: {temp_filename}")
                except Exception as temp_file_error:
                    st.error(f"Error creating temporary file for {name}: {temp_file_error}")
                    any_upload_failed = True
                    continue # Skip this file if temp creation fails

                # Reopen in binary mode for upload
                try:
                    with open(temp_filename, "rb") as f_bin:
                        # Use the upload_and_poll method to ensure file processing completes
                        st.write(f"  - Uploading {name} to Vector Store {self.vector_store_id}...")

                        # ---!!! CORRECTED LINE: Removed .beta !!!---
                        file_batch = self.openai_client.vector_stores.file_batches.upload_and_poll(
                            vector_store_id=self.vector_store_id, files=[f_bin]
                        )
                        # ---!!! END CORRECTION !!!---

                        # Check status
                        if file_batch.status == 'completed':
                            # upload_and_poll response structure might vary.
                            # Check file_counts if available. If completed > 0, assume success for this file.
                            completed_count = 0
                            if hasattr(file_batch, 'file_counts') and file_batch.file_counts:
                                completed_count = file_batch.file_counts.completed

                            if completed_count > 0:
                                uploaded_file_ids.append(name) # Track successful upload by filename
                                st.write(f"    - Upload successful for {name}. Status: {file_batch.status}")
                            else:
                                # If status is completed but count is 0, something might be odd, but treat as uploaded based on status.
                                uploaded_file_ids.append(name)
                                st.write(f"    - Upload reported complete for {name}, but completed count was 0. Status: {file_batch.status}")

                        else:
                            st.warning(f"File batch processing for {name} did not complete successfully. Status: {file_batch.status}")
                            # Log details if available
                            failed_count = file_batch.file_counts.failed if hasattr(file_batch, 'file_counts') else 'N/A'
                            in_progress_count = file_batch.file_counts.in_progress if hasattr(file_batch, 'file_counts') else 'N/A'
                            st.warning(f"  - Details: Failed: {failed_count}, In Progress: {in_progress_count}")
                            any_upload_failed = True


                except Exception as upload_error:
                    st.error(f"Error uploading {name} to vector store {self.vector_store_id}: {upload_error}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    any_upload_failed = True


            # Clean up temporary files regardless of upload success
            for temp_file in temp_files_to_clean:
                 try:
                     os.remove(temp_file)
                 except Exception as cleanup_error:
                     st.warning(f"Could not remove temporary file {temp_file}: {cleanup_error}")


            # Check if *any* upload failed or if the list of successful uploads is empty
            if any_upload_failed or not uploaded_file_ids:
                 st.error("One or more files failed to upload, or no files were successfully added. Vector Store may be incomplete or unusable.")
                 # Attempt to delete the potentially incomplete vector store
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

            # If we reach here, all non-empty files attempted were uploaded successfully
            self.vector_store_created = True
            st.success(f"Vector store created successfully (ID: {self.vector_store_id}) with {len(uploaded_file_ids)} file(s): {', '.join(uploaded_file_ids)}.")
            return True

        except Exception as e:
            st.error(f"Critical Error during vector store creation: {e}")
            st.error(f"Traceback: {traceback.format_exc()}")
            self.vector_store_created = False
            # Clean up vector store if creation failed partially (e.g., before file upload loop)
            if self.vector_store_id:
                try:
                    st.write(f"Attempting cleanup of partially created vector store {self.vector_store_id}...")
                    delete_response = self.openai_client.vector_stores.delete(self.vector_store_id)
                    if delete_response.deleted:
                         st.info(f"Cleaned up partially created vector store {self.vector_store_id}.")
                    else:
                         st.warning(f"Cleanup confirmation failed for vector store {self.vector_store_id}.")
                except Exception as delete_error:
                    st.warning(f"Failed to clean up vector store {self.vector_store_id} after error: {delete_error}")
            self.vector_store_id = None
            return False

    def get_vector_store_results(self, essay):
        """Perform a vector store search (using OpenAI) to get relevant context for grading"""
        if not self.openai_client_initialized or not self.vector_store_id:
             # Return None and False, indicating RAG context is unavailable
             return None, False
        try:
             # Create a search query based on the essay content
             preview = essay[:300] if len(essay) > 300 else essay # Slightly longer preview
             search_query = (
                  f"Identify the 3-5 most relevant criteria or points from the provided rubric and reference material "
                  f"for evaluating an essay starting with this excerpt: '{preview}...' \n"
                  f"Focus on aspects directly applicable to grading this specific essay snippet based on the store's content."
             )

             # --- Using Chat Completion as Retrieval Proxy ---
             # This simulates retrieval if direct vector search is problematic or unavailable.
             # Consider exploring direct vector_stores.search API if needed in the future.
             st.write(f"  - Attempting RAG context retrieval for essay preview via Chat Completion proxy...")
             retrieval_response = self.openai_client.chat.completions.create(
                  model="gpt-4o-mini", # Use a cost-effective, capable model
                  messages=[
                       {"role": "system", "content": "You are an assistant accessing a vector store containing an essay prompt, grading rubric, and reference material. Your task is to retrieve the most relevant context snippets for grading a specific essay based on its initial text."},
                       {"role": "user", "content": search_query}
                  ],
                  temperature=0.0, # Deterministic retrieval
                  max_tokens=300 # Limit token usage for retrieval step
             )
             formatted_results = retrieval_response.choices[0].message.content
             rag_success = bool(formatted_results and formatted_results.strip() and "error" not in formatted_results.lower() and "unable to" not in formatted_results.lower())

             if rag_success:
                  st.write("    - RAG context retrieval successful.")
             else:
                  st.write("    - RAG context retrieval returned no specific information or indicated an issue.")
                  st.write(f"      - Raw retrieval proxy response: {formatted_results[:150]}...") # Log snippet

             return formatted_results, rag_success
             # --- End Retrieval Proxy ---

        except Exception as e:
            st.warning(f"Vector store search/retrieval simulation had an issue: {e}")
            st.warning(f"Traceback: {traceback.format_exc()}")
            return None, False # Return None context and False status on error


    def grade_essay_with_openai(self, model, prompt, rubric, reference, essay, temperature=0.1, top_p=0.9):
        """Grade an essay using an OpenAI model with retrieved context"""
        if not self.openai_client_initialized:
            return {
                "content": "Error: OpenAI client not initialized.",
                "vector_store_created": self.vector_store_created,
                "rag_success": False
            }
        try:
            # Try to get relevant context from vector store (via OpenAI proxy method)
            vector_store_context, rag_success = self.get_vector_store_results(essay)

            # Prepare the grading request
            grading_request_parts = [
                f"Prompt:\n{prompt}\n",
                f"Reference Material:\n{reference}\n",
                f"Rubric:\n{rubric}\n"
            ]
            if rag_success and vector_store_context: # Ensure context is valid and RAG succeeded
                grading_request_parts.append(f"Relevant Context Retrieved:\n{vector_store_context}\n")
            else:
                 grading_request_parts.append("Relevant Context Retrieved: None (or retrieval failed)\n") # Indicate no context was used

            grading_request_parts.extend([
                 f"\nEssay to grade:\n{essay}\n\n",
                 "--- TASK ---",
                 "You are an expert essay grader focusing on detailed, rubric-based feedback.",
                 "Grade this essay according to the rubric and reference material provided. ",
                 "Use the 'Relevant Context Retrieved' section (if provided and relevant) to focus your evaluation. ",
                 "Be specific about points deducted and explain *why*, referencing the rubric criteria directly. ",
                 "Provide a detailed analysis first. Then, at the very end, on a new line, summarize with ONLY the format 'Total Points Deducted: X' where X is a number (e.g., 'Total Points Deducted: 5' or 'Total Points Deducted: 2.5')."
             ])
            grading_request = "".join(grading_request_parts)


            # Base request parameters
            request_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert essay grader focusing on detailed, rubric-based feedback, adhering strictly to the provided rubric, reference, and retrieved context."},
                    {"role": "user", "content": grading_request}
                ],
                 "max_tokens": 4000 # Ensure enough space for detailed feedback
            }

            # Apply temperature and top_p only if the model/provider supports them
            provider_config = MODEL_OPTIONS.get("OpenAI", {})
            if provider_config.get("supports_temperature", False):
                 # Adjust temperature within the provider's allowed range
                 min_temp, max_temp = provider_config.get("temp_range", (0.0, 1.0))
                 request_params["temperature"] = max(min_temp, min(temperature, max_temp))
            if provider_config.get("supports_top_p", False):
                 request_params["top_p"] = top_p

            response = self.openai_client.chat.completions.create(**request_params)

            # Return content along with RAG status flags
            return {
                "content": response.choices[0].message.content,
                "vector_store_created": self.vector_store_created, # Status of the store creation attempt
                "rag_success": rag_success # Status of retrieving context for *this* essay
            }

        except Exception as e:
            st.error(f"Error grading with OpenAI model {model}: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
            return {
                "content": f"Error grading with OpenAI model {model}: {str(e)}",
                "vector_store_created": self.vector_store_created,
                "rag_success": False # RAG failed if there was an error
            }

    def grade_essay_with_anthropic(self, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        """Grade an essay using an Anthropic model"""
        # NOTE: Anthropic calls *cannot* directly use the OpenAI vector store API.
        # We *reuse* the context retrieved via the OpenAI mechanism (`get_vector_store_results`).
        try:
            # Try to get relevant context from the OpenAI vector store proxy
            vector_store_context, rag_success = self.get_vector_store_results(essay)

            # Prepare the grading request (similar structure to OpenAI)
            grading_request_parts = [
                 f"Prompt:\n{prompt}\n",
                 f"Reference Material:\n{reference}\n",
                 f"Rubric:\n{rubric}\n"
             ]
            if rag_success and vector_store_context:
                grading_request_parts.append(f"Relevant Context Retrieved (from external system):\n{vector_store_context}\n")
            else:
                grading_request_parts.append("Relevant Context Retrieved: None (or retrieval failed)\n")

            grading_request_parts.extend([
                 f"\nEssay to grade:\n{essay}\n\n",
                 "--- TASK ---",
                 "You are an expert essay grader focusing on detailed, rubric-based feedback.",
                 "Grade this essay according to the rubric and reference material provided. ",
                 "Use the 'Relevant Context Retrieved' section (if provided and relevant) to focus your evaluation. ",
                 "Be specific about points deducted and explain *why*, referencing the rubric criteria directly. ",
                 "Provide a detailed analysis first. Then, at the very end, on a new line, summarize with ONLY the format 'Total Points Deducted: X' where X is a number (e.g., 'Total Points Deducted: 5' or 'Total Points Deducted: 2.5')."
             ])
            # Combine system prompt and user request for Anthropic
            system_prompt_anthropic = "You are an expert essay grader focusing on detailed, rubric-based feedback, adhering strictly to the provided rubric, reference, and retrieved context."
            user_prompt_anthropic = "".join(grading_request_parts)


            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key, # Use the stored API key
                "anthropic-version": "2023-06-01" # Or latest recommended version like "2024-01-01"
            }

            # Base data payload
            data = {
                "model": model,
                "system": system_prompt_anthropic, # Use the system parameter
                "messages": [
                     {"role": "user", "content": user_prompt_anthropic}
                ],
                "max_tokens": 4000 # Claude models support large context windows
            }


            # Apply temperature and top_p based on provider config
            provider_config = MODEL_OPTIONS.get("Anthropic", {})
            if provider_config.get("supports_temperature", False):
                 min_temp, max_temp = provider_config.get("temp_range", (0.0, 1.0))
                 data["temperature"] = max(min_temp, min(temperature, max_temp))
            if provider_config.get("supports_top_p", False):
                 # Ensure top_p is not None before adding
                 if top_p is not None:
                      data["top_p"] = top_p

            response = requests.post("https://api.anthropic.com/v1/messages", json=data, headers=headers)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            response_data = response.json()

            # Check for errors or unexpected structure in response
            if response_data.get("type") == "error":
                error_details = response_data.get("error", {})
                raise ValueError(f"Anthropic API Error: Type: {error_details.get('type', 'N/A')}, Message: {error_details.get('message', 'No message provided')}")
            if "content" not in response_data or not isinstance(response_data["content"], list) or len(response_data["content"]) == 0:
                 raise ValueError("Unexpected response format from Anthropic API: 'content' missing or invalid.")

            # Extract text, checking for correct type
            if response_data["content"][0].get("type") == "text":
                content = response_data["content"][0]["text"]
            else:
                raise ValueError("Expected text content from Anthropic API, but received different type.")


            return {
                "content": content,
                "vector_store_created": self.vector_store_created, # Reflects OpenAI store status
                "rag_success": rag_success # Reflects OpenAI RAG status
            }

        except requests.exceptions.RequestException as e:
             # Handle network/HTTP errors
             error_content = f"API Request Error: {str(e)}"
             if e.response is not None:
                  error_content += f"\nStatus Code: {e.response.status_code}"
                  try:
                       error_detail = e.response.json()
                       error_content += f"\nDetails: {json.dumps(error_detail)}"
                  except json.JSONDecodeError:
                       error_content += f"\nResponse Body (non-JSON): {e.response.text}"
             st.error(f"Error grading with Anthropic model {model}: {error_content}")
             st.error(f"Traceback: {traceback.format_exc()}")
             return {
                  "content": f"Error grading with Anthropic model {model}: {error_content}",
                  "vector_store_created": self.vector_store_created,
                  "rag_success": False
             }

        except Exception as e:
             # Handle other errors (e.g., response parsing, ValueError)
             st.error(f"Error grading with Anthropic model {model}: {str(e)}")
             st.error(f"Traceback: {traceback.format_exc()}")
             return {
                "content": f"Error grading with Anthropic model {model}: {str(e)}",
                "vector_store_created": self.vector_store_created,
                "rag_success": False
             }

    # --- Method for Google Gemini ---
    def grade_essay_with_google(self, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        """Grade an essay using a Google Gemini model"""
        # NOTE: Gemini calls *cannot* directly use the OpenAI vector store API.
        # We *reuse* the context retrieved via the OpenAI mechanism (`get_vector_store_results`).
        try:
            # Configure the Gemini client (needs to be done before each use potentially, or once if key is confirmed valid)
            try:
                 genai.configure(api_key=self.api_key)
            except Exception as config_error:
                 st.error(f"Error configuring Google Gemini client: {str(config_error)}")
                 return {
                      "content": f"Error configuring Google Gemini client: {str(config_error)}",
                      "vector_store_created": self.vector_store_created,
                      "rag_success": False
                 }

            # Try to get relevant context from the OpenAI vector store proxy
            vector_store_context, rag_success = self.get_vector_store_results(essay)

            # Prepare the grading request (similar structure)
            # Use a clear system prompt approach suitable for Gemini
            system_instruction = (
                "You are an expert essay grader focusing on detailed, rubric-based feedback. "
                "Adhere strictly to the provided rubric, reference material, and retrieved context."
            )

            user_prompt_parts = [
                 f"**Prompt:**\n{prompt}\n",
                 f"**Reference Material:**\n{reference}\n",
                 f"**Rubric:**\n{rubric}\n"
             ]
            if rag_success and vector_store_context:
                user_prompt_parts.append(f"**Relevant Context Retrieved (from external system):**\n{vector_store_context}\n")
            else:
                user_prompt_parts.append("**Relevant Context Retrieved:** None (or retrieval failed)\n")

            user_prompt_parts.extend([
                 f"\n**Essay to grade:**\n{essay}\n\n",
                 "--- **TASK** ---",
                 "1. Grade this essay according to the rubric and reference material provided. ",
                 "2. Use the 'Relevant Context Retrieved' section (if provided and relevant) to focus your evaluation. ",
                 "3. Be specific about points deducted and explain *why*, referencing the rubric criteria directly. ",
                 "4. Provide a detailed analysis first.",
                 "5. Then, at the very end, on a new line, summarize with ONLY the format 'Total Points Deducted: X' where X is a number (e.g., 'Total Points Deducted: 5' or 'Total Points Deducted: 2.5')."
             ])
            full_user_prompt = "".join(user_prompt_parts)

            # Initialize the Gemini model with system instruction if supported by the model/API version
            # Newer Gemini APIs might prefer system_instruction parameter
            try:
                gemini_model = genai.GenerativeModel(
                    model,
                    system_instruction=system_instruction # Pass system instruction here
                )
            except TypeError: # Handle older versions that might not accept system_instruction in constructor
                gemini_model = genai.GenerativeModel(model)
                # Prepend system instruction to the user prompt if needed (less ideal)
                # full_user_prompt = system_instruction + "\n\n" + full_user_prompt
                st.warning("Gemini model/SDK version might not support 'system_instruction' directly in constructor. Functionality might be slightly affected.")


            # Apply temperature and top_p based on provider config
            provider_config = MODEL_OPTIONS.get("Google", {})
            gen_config_params = {}
            if provider_config.get("supports_temperature", False):
                 min_temp, max_temp = provider_config.get("temp_range", (0.0, 1.0))
                 # Clamp temperature to Google's typical max (often 1.0, sometimes 2.0)
                 effective_max_temp = min(max_temp, 1.0) # Be conservative, use 1.0 unless known otherwise
                 gen_config_params["temperature"] = max(min_temp, min(temperature, effective_max_temp))
            if provider_config.get("supports_top_p", False):
                 # Ensure top_p is not None before adding
                 if top_p is not None:
                      gen_config_params["top_p"] = top_p # Gemini generally supports top_p

            generation_config = genai.types.GenerationConfig(**gen_config_params)

             # Define safety settings (adjust as needed - BLOCK_MEDIUM_AND_ABOVE is a reasonable default)
            safety_settings = [
                 {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             ]


            # Make the API call
            response = gemini_model.generate_content(
                 full_user_prompt, # Send the combined user prompt
                 generation_config=generation_config,
                 safety_settings=safety_settings # Apply safety settings
                 )

            # Check for safety blocks or other issues in the response
            try:
                 # Accessing response.text might raise an exception if blocked or no content
                 content = response.text
            except ValueError as e:
                 # This often indicates blocking due to safety filters or lack of content
                 block_reason = "Unknown"
                 safety_feedback_str = "N/A"
                 finish_reason = "Unknown"
                 try:
                     if response.prompt_feedback:
                          block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback.block_reason else "Not Specified"
                          safety_feedback_str = str(response.prompt_feedback.safety_ratings) # Get ratings list
                 except (AttributeError, ValueError):
                     pass # No prompt_feedback attribute or issues accessing it

                 try:
                      # Check candidate finish reason if available
                      if response.candidates and response.candidates[0].finish_reason:
                          finish_reason = response.candidates[0].finish_reason.name
                 except (AttributeError, IndexError, ValueError):
                      pass

                 content = (f"Content generation issue. Finish Reason: {finish_reason}.\n"
                            f"Safety Block Reason: {block_reason}.\n"
                            f"Safety Feedback Ratings: {safety_feedback_str}\n"
                            f"Original error hint: {str(e)}")
                 rag_success = False # If blocked, RAG context wasn't effectively used
                 st.warning(f"Gemini content generation issue for model {model}. Finish: {finish_reason}, Block: {block_reason}")
            except Exception as e: # Catch other potential errors during text access
                 content = f"Error accessing Gemini response content: {str(e)}"
                 rag_success = False
                 st.error(f"Error accessing Gemini response text for model {model}: {e}")


            return {
                "content": content,
                "vector_store_created": self.vector_store_created, # Reflects OpenAI store status
                "rag_success": rag_success # Reflects OpenAI RAG status (or False if blocked/error)
            }

        except Exception as e:
            st.error(f"Error grading with Google model {model}: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
            return {
                "content": f"Error grading with Google model {model}: {str(e)}",
                "vector_store_created": self.vector_store_created,
                "rag_success": False
            }
    # --- End Google Gemini Method ---

    def grade_essay(self, provider, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        """Route the grading request to the appropriate API based on provider"""
        if provider == "OpenAI":
            return self.grade_essay_with_openai(model, prompt, rubric, reference, essay, temperature, top_p)
        elif provider == "Anthropic":
            # Pass both temp and top_p as Anthropic API supports them
            return self.grade_essay_with_anthropic(model, prompt, rubric, reference, essay, temperature, top_p)
        elif provider == "Google":
             return self.grade_essay_with_google(model, prompt, rubric, reference, essay, temperature, top_p)
        else:
            st.error(f"Attempted to grade with unsupported provider: {provider}")
            return {
                "content": f"Unsupported provider: {provider}",
                "vector_store_created": False,
                "rag_success": False
            }


def save_pdf(essay_name, model_outputs):
    """Generate a PDF with grading results and RAG status using standard fonts"""
    try:
        # Create a temporary directory to store the PDF
        temp_dir = tempfile.mkdtemp()
        # Sanitize essay name for filename more thoroughly
        safe_essay_name_for_file = ''.join(c for c in essay_name if c.isalnum() or c in ('_', '-')).strip()
        if not safe_essay_name_for_file:
             safe_essay_name_for_file = "graded_essay"
        # Ensure max length for filename
        safe_essay_name_for_file = safe_essay_name_for_file[:100]
        temp_pdf_path = os.path.join(temp_dir, f"{safe_essay_name_for_file}.pdf")

        # Create PDF with standard fonts only (Arial, Times, Courier)
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # --- Title ---
        pdf.set_font('Arial', 'B', 14)
        # Encode title safely for Latin-1 used by FPDF's core fonts ('replace' bad chars)
        safe_title_display = essay_name.encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(0, 10, f"Graded Essay: {safe_title_display}", ln=True, align="C")
        pdf.ln(5) # Add some space

        # Process each model's output
        for model, output_data in model_outputs.items():
            # --- Model Header ---
            pdf.set_font('Arial', 'B', 12)
            safe_model_display = model.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 10, f"--- Model: {safe_model_display} ---", ln=True)

            # --- RAG Status ---
            rag_status = (f"Vector Store: {'Created' if output_data.get('vector_store_created', False) else 'Not Created/Failed'} | "
                           f"RAG Context: {'Retrieved' if output_data.get('rag_success', False) else 'Not Retrieved/Failed'}")
            pdf.set_font('Arial', 'I', 9) # Italic smaller font for status
            # Encode status safely
            safe_rag_status = rag_status.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 8, f"Status: {safe_rag_status}", ln=True)
            pdf.ln(2) # Space before content

            # --- Graded Content ---
            pdf.set_font('Arial', '', 10) # Use Arial as a common fallback
            content = output_data.get("content", "Error: No content found.")
            # Process text line by line with encoding safety for PDF core fonts
            for line in content.split("\n"):
                 # Replace non-Latin-1 characters with '?' for FPDF standard fonts
                 safe_line = line.encode('latin-1', 'replace').decode('latin-1')
                 pdf.multi_cell(0, 5, safe_line) # Use multi_cell for line wrapping

            pdf.ln(5) # Add space between model outputs

        # Output to temporary file
        pdf.output(temp_pdf_path)

        # Read the file into memory
        with open(temp_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        # Clean up the temporary directory and file
        try:
            os.remove(temp_pdf_path)
            os.rmdir(temp_dir)
        except Exception as cleanup_error:
             st.warning(f"Could not clean up temporary PDF file/directory: {cleanup_error}")

        return pdf_bytes

    except Exception as e:
        st.warning(f"PDF generation failed for '{essay_name}': {e}. Results will be in TXT/CSV only.")
        st.warning(f"Traceback: {traceback.format_exc()}")
        # If PDF generation fails, return None
        return None


def display_comparison_table(results):
    """Display results table and return dataframe for export"""
    try:
        total_scores = []
        rows = []

        for model, output_data in results.items():
            content = output_data.get("content", "Error: No content available.")
            # Ensure content is a string before splitting
            if not isinstance(content, str):
                content = str(content)

            vector_store_created = output_data.get("vector_store_created", False)
            rag_success = output_data.get("rag_success", False)

            # Extract points deducted and summary more robustly
            lines = content.splitlines()
            deduction_line_text = "Not Found"
            points = None # Use None to indicate not found/parsed

            # Search from the end for the deduction line for better chance of finding summary
            for line in reversed(lines):
                 line_lower = line.lower().strip()
                 # Look for specific pattern, case-insensitive, strict format
                 if line_lower.startswith("total points deducted:"):
                      deduction_line_text = line.strip()
                      # Try to extract numerical value safely (float or int)
                      try:
                           # More robust extraction - find number after colon, potentially with spaces
                           numeric_part_str = line.split(":")[-1].strip()
                           # Remove any non-numeric characters except '.' and '-' (for potential negative deductions, though unlikely)
                           # Allow only digits, one decimal point, and an optional leading minus sign
                           cleaned_numeric_str = ''.join(c for c in numeric_part_str if c.isdigit() or c == '.' or (c == '-' and numeric_part_str.startswith('-')))
                           # Ensure it's a valid number format
                           if cleaned_numeric_str and cleaned_numeric_str != '-' and cleaned_numeric_str != '.':
                                points = float(cleaned_numeric_str)
                      except ValueError:
                           points = None # Failed to parse
                      except Exception: # Catch other potential issues
                          points = None
                      break # Found the target line, stop searching

            # Get a summary (e.g., first few lines or last non-empty line before deduction)
            summary = ""
            non_empty_lines = [l.strip() for l in lines if l.strip() and not l.lower().strip().startswith("total points deducted:")]
            if non_empty_lines:
                 # Take the first ~3 non-empty lines as a summary snippet
                 summary = "\n".join(non_empty_lines[:3])
                 if len(non_empty_lines) > 3:
                      summary += "..."


            if points is not None:
                total_scores.append((model, points))

            # Add RAG status to the table
            rag_status_str = (f"Store: {'Created' if vector_store_created else 'Not Created/Failed'} | "
                               f"RAG: {'Retrieved' if rag_success else 'Not Retrieved/Failed'}")

            rows.append({
                "Model": model,
                "RAG Status": rag_status_str,
                "Points Deducted Info": deduction_line_text, # Show the raw line found
                "Extracted Points": f"{points:.1f}" if points is not None else "N/A", # Formatted points
                "Feedback Summary": summary[:250] + ('...' if len(summary)>250 else '') # Limit summary length slightly more
            })

        # Create and display the dataframe
        if not rows:
            st.warning("No results to display in the table.")
            return None

        df = pd.DataFrame(rows)
        # Define column order for clarity
        df = df[["Model", "RAG Status", "Extracted Points", "Points Deducted Info", "Feedback Summary"]]
        st.dataframe(df, use_container_width=True)

        if total_scores:
            st.subheader("📊 Average Points Deducted (Successfully Parsed)")
            score_df = pd.DataFrame(total_scores, columns=["Model", "Points Deducted"])
            # Calculate mean, handling potential division by zero if no scores for a model
            # Also calculate standard deviation for variability insight
            avg_std = score_df.groupby("Model")["Points Deducted"].agg(['mean', 'std', 'count']).reset_index()
            avg_std.rename(columns={'mean': 'Average Deduction', 'std': 'Std Dev Deduction', 'count': 'Count'}, inplace=True)
            # Format floats
            avg_std['Average Deduction'] = avg_std['Average Deduction'].map('{:.2f}'.format)
            avg_std['Std Dev Deduction'] = avg_std['Std Dev Deduction'].map('{:.2f}'.format)
            st.table(avg_std)

        # Return dataframe for CSV export
        return df
    except Exception as e:
        st.error(f"Error displaying comparison table: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


def main():
    st.set_page_config(page_title="Multi-Model Essay Grader", layout="wide")
    st.title("🤖 Multi-Model Essay Grader with Vector Search")
    st.caption(f"Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}") # Added Timezone

    # Instructions Expander
    with st.expander("💡 Instructions & Setup"):
        st.markdown("""
**Goal:** Grade essays using multiple AI models (OpenAI, Anthropic, Google Gemini) and compare outputs, potentially enhancing consistency with Retrieval-Augmented Generation (RAG) using OpenAI's Vector Store.

**Using the App Interface (Step-by-Step):**

1.  **Upload API Key:** Locate the first file uploader labeled **`🔑 API Key (TXT)`**. Click "Browse files" (or drag and drop) to upload the TXT file containing your API key.
    * **Important:** This tool currently uses the *same* uploaded key for *all* selected providers (OpenAI, Anthropic, Google). Ensure the key is valid for the services you choose.
2.  **Upload Prompt:** Use the uploader labeled **`📝 Prompt (TXT)`** to upload your essay prompt file.
3.  **Upload Rubric:** Use the uploader labeled **`📋 Rubric (TXT)`** to upload your grading rubric file.
4.  **Upload Reference Material:** Use the uploader labeled **`📘 Reference (TXT)`** to upload any reference material file. *(Ensure context files are not empty and use UTF-8 encoding).*
5.  **Upload Essays:**
    * Go to the section **`📂 Upload Essays`**.
    * **Option A (ZIP):** Use the uploader **`📦 Upload ZIP of Essays (TXT files, UTF-8)`** to upload a single ZIP file containing all your anonymized TXT essay files.
    * **Option B (Individual Files):** If not using a ZIP, use the uploader **`📂 Or Upload one or more Essay Files (TXT, UTF-8)`** to select and upload multiple individual anonymized TXT essay files.
    * *(Ensure essays are anonymized and UTF-8 encoded).*
    * *Check the info message below the uploaders to confirm essays loaded.*
6.  **Select Models:**
    * Scroll down to the section **`🤖 Choose Models for Grading`**.
    * Click on the expander arrows (e.g., `OpenAI Models`, `Google Models`) to see the available models.
    * Check the boxes next to the specific models you want to use. *(Note: Preview models might have different availability or limits).*
7.  **Adjust Settings (Optional):**
    * Go to the section **`⚙️ Model Behavior Settings`**.
    * Adjust the **`Temperature`** and **`Top-p Sampling`** sliders if needed. *(These only affect models that support them).*
8.  **Start Grading:** Click the large button labeled **`🚀 Grade Essays`**.
9.  **Wait for Processing:** Monitor the progress bars and status text as the app processes each essay and model. This can take some time, especially with many essays or models. Look for status messages about Vector Store creation and RAG context retrieval.
10. **Review Results:** As processing completes for each essay, results will appear below. Review the:
    * **`RAG Status`** table (shows Vector Store/Context Retrieval status per model for *that* essay).
    * **`Comparison Table`** (summarizes feedback, points deducted, RAG status, etc.).
    * **`Average Points Deducted`** table (includes average, standard deviation, and count of successfully parsed scores).
11. **Download Outputs:**
    * Use the **`📥 Download Results for '[essay_name]' (ZIP)`** button below each essay's results section for individual downloads (TXT, PDF, CSV combined).
    * After *all* essays are done, use the **`📦 Download ALL Results (Single ZIP)`** button at the very bottom for a combined download of all results.

**Understanding the Process (Behind the Scenes):**

* The tool attempts to create an **OpenAI Vector Store** using your uploaded Context Files (Prompt, Rubric, Reference). This store helps find relevant grading information. This step happens once per run.
* For each essay and selected model:
    * It attempts to retrieve relevant context from the Vector Store using a **Chat Completion Proxy** (this is the **RAG via OpenAI** step).
    * It sends the essay, original prompt, rubric, reference, and the *retrieved context* (if successful) to the chosen model's API (OpenAI, Anthropic, or Google).
    * The **RAG Status** displayed reflects the success of the OpenAI Vector Store creation (once at the start) and context retrieval (per essay); all models benefit from successfully retrieved context via the prompt.
        """)

    # About Expander
    with st.expander("📚 About & Disclaimer"):
        st.markdown(
            """
**Multi-Model Essay Grader with Retrieval-Augmented Generation**

Experiment with AI-assisted grading using OpenAI, Anthropic, and Google Gemini models. This tool leverages OpenAI's vector stores for retrieval-augmented generation (RAG) to potentially improve grading consistency by providing relevant rubric/reference context to the chosen AI model.

**Key Features:**
- **Multi-Provider Support:** Compare OpenAI, Anthropic, and Google Gemini models.
- **Vector Store RAG (via OpenAI):** Uses OpenAI's service to create a vector store from your prompt, rubric, and reference. Attempts to retrieve relevant context (RAG) for each essay using a chat completion proxy.
- **Context Injection:** The retrieved context (if any) is added to the prompt sent to *all* selected models (OpenAI, Anthropic, Google).
- **Comprehensive Output:** Detailed feedback, points deducted (if parsable), and RAG status.
- **Consolidated Results:** Download results in TXT, PDF (basic formatting), and CSV formats, packaged per essay or as a single ZIP for all.

**How Vector Storage & RAG Work Here:**
1.  An OpenAI Vector Store is created *once* at the beginning of a run using your uploaded prompt, rubric, and reference texts. Status messages will indicate success or failure.
2.  When grading *each* essay, the system attempts to query this OpenAI store (via a chat proxy) to find relevant sections based on the essay's content.
3.  This retrieved context (RAG result) is then included in the instructions sent to the selected OpenAI, Anthropic, or Google model *for that essay*.
4.  The 'RAG Status' in the results table reflects the success of the initial OpenAI vector store creation and the context retrieval *for that specific essay*.

**Data Privacy:**
- Files are processed in memory during your session. No data is stored server-side by this application.
- Vector stores created via OpenAI exist only for the session duration (or until potentially deleted by OpenAI policies if left orphaned). The app attempts to delete stores if uploads fail.
- Data (prompt, rubric, reference, essay, retrieved context) is sent to the API providers (OpenAI, Anthropic, Google) you select. Consult their respective privacy policies.
- **Anonymize student data (essays) before uploading.**

**Important Notes:**
- **API Key:** The tool currently uses one API key for all selected providers. Ensure compatibility or select providers accordingly.
- **Evaluation Tool:** This is for exploring AI grading, not replacing human judgment. Educators must verify AI output and assign final grades.
- **Preview Models:** Models marked "Preview" may have limitations or change.
- **Compliance:** Users must comply with privacy laws (e.g., FERPA, GDPR). Anonymize data.
- **PDF Output:** Uses basic fonts; special characters might be replaced.

**License & Author:**
Created by Jonathan Graziola (isidore.gpt@gmail.com), modified by Gemini. Licensed under GNU GPL v3.0.

**Disclaimer:** Use responsibly and ethically. The tool and its outputs come with NO WARRANTY.
            """
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        api_key_file = st.file_uploader("🔑 API Key (TXT)", type=["txt"], help="Upload a TXT file with your API key (used for all selected providers).")
    with col2:
        prompt_file = st.file_uploader("📝 Prompt (TXT)", type=["txt"])
    with col3:
        rubric_file = st.file_uploader("📋 Rubric (TXT)", type=["txt"])
    with col4:
        reference_file = st.file_uploader("📘 Reference (TXT)", type=["txt"])

    st.markdown("---")
    st.subheader("📂 Upload Essays")
    essay_zip = st.file_uploader("📦 Upload ZIP of Essays (TXT files, UTF-8)", type=["zip"])

    essay_files = []
    essay_names = []

    # Robust file handling with error messages
    if essay_zip:
        try:
            with zipfile.ZipFile(essay_zip) as z:
                all_files = z.namelist()
                # Filter TXT, ignore MacOSX metadata and hidden files/folders
                essay_names_in_zip = [
                    f for f in all_files
                    if f.endswith(".txt") and
                    not f.startswith('__MACOSX/') and
                    not os.path.basename(f).startswith('.') and
                    '/' not in f # Basic check to avoid files in subdirectories within zip
                 ]
                if not essay_names_in_zip:
                     st.warning("No '.txt' files found directly in the root of the uploaded ZIP.")
                temp_essay_files = []
                temp_essay_names = []
                for f_name in essay_names_in_zip:
                     try:
                         # Use 'replace' for decoding errors, common with student files
                         file_content = z.read(f_name).decode("utf-8", errors="replace")
                         if file_content.strip(): # Only add non-empty files
                            temp_essay_files.append(file_content)
                            temp_essay_names.append(os.path.basename(f_name)) # Store only filename
                         else:
                             st.warning(f"Skipping empty file '{f_name}' from ZIP.")
                     except Exception as decode_error:
                          st.error(f"Error decoding file '{f_name}' from ZIP: {decode_error}. Skipping this file.")
                essay_files = temp_essay_files
                essay_names = temp_essay_names
        except zipfile.BadZipFile:
            st.error("Invalid or corrupted ZIP file.")
            essay_zip = None # Reset zip file state
        except Exception as e:
            st.error(f"Error reading ZIP file: {e}")
            essay_zip = None
    else:
        uploaded_files = st.file_uploader("📂 Or Upload one or more Essay Files (TXT, UTF-8)", type=["txt"], accept_multiple_files=True)
        if uploaded_files:
            temp_essay_files = []
            temp_essay_names = []
            for f in uploaded_files:
                try:
                    # Use 'replace' for decoding errors
                    file_content = f.read().decode("utf-8", errors="replace")
                    if file_content.strip(): # Only add non-empty files
                        temp_essay_files.append(file_content)
                        temp_essay_names.append(f.name)
                    else:
                        st.warning(f"Skipping empty uploaded file: {f.name}")
                except Exception as e:
                    st.error(f"Error reading file {f.name}: {e}. Skipping this file.")
            essay_files = temp_essay_files
            essay_names = temp_essay_names

    if essay_files:
         st.info(f"Loaded {len(essay_files)} non-empty essay(s): {', '.join(essay_names)}")
    else:
         st.info("No essays loaded yet.")


    st.markdown("---")
    st.subheader("🤖 Choose Models for Grading")
    selected_models = []
    # Dynamically create checkboxes based on MODEL_OPTIONS
    provider_columns = st.columns(len(MODEL_OPTIONS))
    col_index = 0
    for provider, provider_config in MODEL_OPTIONS.items():
        with provider_columns[col_index]:
             st.markdown(f"**{provider} Models**")
             # Removed expander for flatter layout
             # with st.expander(f"{provider} Models ({len(provider_config['models'])} available)"):
             for model in provider_config["models"]:
                 # Use a unique key for each checkbox
                 is_preview = "preview" in model.lower() or "latest" in model.lower() # Also consider 'latest' potentially preview-like
                 label = f"{model}{' (Preview/Latest)' if is_preview else ''}"
                 if st.checkbox(label, key=f"model_{provider}_{model}"):
                     selected_models.append({"provider": provider, "model": model})
        col_index += 1


    if selected_models:
         st.write("Selected Models:", [f"{m['provider']} - {m['model']}" for m in selected_models])
    else:
        st.warning("No models selected.")

    st.markdown("---")
    st.subheader("⚙️ Model Behavior Settings")
    st.info("Note: Settings are applied only to models that support them according to configuration. Check provider docs for specifics.")

    # Use columns for sliders
    col_temp, col_top_p = st.columns(2)
    with col_temp:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, # Default 0.2, clamped max to 1.0 for broader compatibility
                                 help="Controls randomness (0.0 = deterministic, 1.0 = max creative). Applied where supported.")
    with col_top_p:
        top_p = st.slider("Top-p Sampling", 0.1, 1.0, 0.9, 0.05, # Default 0.9
                           help="Nucleus sampling (considers tokens comprising the top 'p' probability mass). Applied where supported.")

    st.markdown("---")

    if st.button("🚀 Grade Essays", type="primary"):
        # --- Input Validation ---
        valid_inputs = True
        if not api_key_file:
            st.error("❌ Please upload the API Key file.")
            valid_inputs = False
        if not prompt_file:
            st.error("❌ Please upload the Prompt file.")
            valid_inputs = False
        if not rubric_file:
            st.error("❌ Please upload the Rubric file.")
            valid_inputs = False
        if not reference_file:
            st.error("❌ Please upload the Reference Material file.")
            valid_inputs = False
        if not essay_files: # Check if the list is empty after processing uploads
            st.error("❌ Please upload at least one valid, non-empty Essay (TXT file or via ZIP).")
            valid_inputs = False
        if not selected_models:
            st.error("❌ Please select at least one AI Model.")
            valid_inputs = False

        if valid_inputs:
            # --- Process Inputs ---
            try:
                api_key = api_key_file.read().decode("utf-8", errors="replace").strip()
                if not api_key:
                     raise ValueError("API key file is empty.")
                prompt = prompt_file.read().decode("utf-8", errors="replace")
                rubric = rubric_file.read().decode("utf-8", errors="replace")
                reference = reference_file.read().decode("utf-8", errors="replace")

                # Initialize the model manager (which also attempts OpenAI client init)
                model_manager = ModelManager(api_key)

                # --- Create Vector Store (Once per run) ---
                vector_store_created_overall = False # Track if store creation attempt was successful
                if model_manager.openai_client_initialized: # Only attempt if OpenAI client is ready
                     with st.spinner("Attempting to create OpenAI Vector Store for RAG context..."):
                         # Pass context file contents directly
                         vector_store_created_overall = model_manager.create_vector_store(prompt, rubric, reference)
                         # Status messages handled within create_vector_store method now
                         if not vector_store_created_overall:
                             st.warning("⚠️ Failed to create OpenAI Vector Store. RAG context retrieval will be skipped.")
                 else:
                     st.warning("⚠️ OpenAI client failed to initialize. Skipping Vector Store creation and RAG.")


                # --- Grade Each Essay ---
                all_essay_data_for_zip = [] # To store data for final bulk zip

                st.markdown("---") # Separator before results start appearing
                st.header("📊 Grading Results")

                # Main loop through essays
                total_essays = len(essay_files)
                overall_progress_bar = st.progress(0, text="Starting essay processing...")

                for i, essay_text in enumerate(essay_files):
                    essay_name = essay_names[i] if i < len(essay_names) else f"essay_{i+1}.txt"
                    st.subheader(f"Processing Essay {i+1}/{total_essays}: {essay_name}")

                    # Update overall progress
                    overall_progress_bar.progress((i / total_essays), text=f"Processing Essay {i+1}/{total_essays}: {essay_name}")

                    # Basic check for valid essay text (double-check, though filtered earlier)
                    if not isinstance(essay_text, str) or not essay_text.strip():
                         st.warning(f"Skipping '{essay_name}' due to empty or invalid content found during main loop.")
                         continue

                    results_this_essay = {}
                    model_progress_bar = st.progress(0, text=f"Starting model grading for '{essay_name}'...")

                    # Container for RAG status per essay (shown before the table)
                    rag_status_container = st.container()
                    rag_statuses_this_essay = [] # List to hold dicts for dataframe

                    model_count = len(selected_models)
                    for idx, model_info in enumerate(selected_models):
                        provider = model_info["provider"]
                        model = model_info["model"]
                        model_key = f"{provider} - {model}" # Consistent key

                        progress_text = f"Grading '{essay_name}' with {model_key} ({idx+1}/{model_count})..."
                        model_progress_bar.progress((idx + 1) / model_count, text=progress_text)


                        # Determine effective temperature and top_p based on provider support
                        provider_config = MODEL_OPTIONS.get(provider, {})
                        current_temp = temperature # Start with user setting
                        current_top_p = top_p     # Start with user setting

                        # No need to override if not supported, the individual methods handle it
                        # if not provider_config.get("supports_temperature", True):
                        #      current_temp = None # Let API handle default
                        # if not provider_config.get("supports_top_p", True):
                        #     current_top_p = None # Explicitly None if not supported

                        # Grade using the manager
                        try:
                            # Pass the vector store creation status (happened once) to grade_essay
                            result_data = model_manager.grade_essay(
                                provider, model, prompt, rubric, reference, essay_text,
                                temperature=current_temp,
                                top_p=current_top_p # Pass None if not supported by provider config
                            )

                            # Ensure result_data has expected keys even on error returns from grade_essay
                            results_this_essay[model_key] = {
                                "content": result_data.get("content", "Error: Missing content"),
                                "vector_store_created": result_data.get("vector_store_created", model_manager.vector_store_created), # Use manager status as fallback
                                "rag_success": result_data.get("rag_success", False)
                            }

                            # Update RAG status display for this model using the returned data
                            rag_statuses_this_essay.append({
                                 "Model": model_key,
                                 "Vector Store Status (Overall)": "Created" if results_this_essay[model_key]['vector_store_created'] else "Not Created/Failed",
                                 "RAG Context Retrieval (This Essay)": "Retrieved" if results_this_essay[model_key]['rag_success'] else "Not Retrieved/Failed/Skipped"
                            })


                        except Exception as grade_error:
                             st.error(f"Critical error calling grade_essay for {model_key}: {grade_error}")
                             st.error(f"Traceback: {traceback.format_exc()}")
                             # Log failure in results
                             results_this_essay[model_key] = {
                                 "content": f"Error during grading call: {str(grade_error)}",
                                 "vector_store_created": model_manager.vector_store_created, # Best guess
                                 "rag_success": False
                             }
                             # Also update RAG status table to show failure
                             rag_statuses_this_essay.append({
                                 "Model": model_key,
                                 "Vector Store Status (Overall)": "Created" if model_manager.vector_store_created else "Not Created/Failed",
                                 "RAG Context Retrieval (This Essay)": "Failed (Grading Error)"
                             })

                        # Short delay between API calls
                        time.sleep(1.5) # Increase delay slightly to help avoid rate limits

                    # Display RAG status table after all models for this essay are done
                    with rag_status_container:
                         st.markdown("##### RAG Status (per model for this essay):")
                         if rag_statuses_this_essay:
                              st.dataframe(pd.DataFrame(rag_statuses_this_essay), use_container_width=True)
                         else:
                              st.write("No models processed for RAG status.")

                    model_progress_bar.empty() # Clear progress bar after finishing models for this essay
                    st.success(f"Finished processing models for '{essay_name}'.")

                    # --- Display Results for This Essay ---
                    st.markdown("##### Comparison Table:")
                    results_df_this_essay = display_comparison_table(results_this_essay)

                    # --- Prepare Files for Download (This Essay) ---
                    essay_files_for_zip = {}
                    # Sanitize filename again, ensure max length
                    base_filename = ''.join(c for c in essay_name if c.isalnum() or c in ('_', '-')).strip()[:100]
                    if not base_filename: base_filename = f"essay_{i}"

                    # 1. Text file
                    try:
                        text_content = f"ESSAY NAME: {essay_name}\n\n"
                        text_content += f"ORIGINAL ESSAY:\n{'-'*20}\n{essay_text}\n{'-'*20}\n\n"
                        text_content += "GRADING RESULTS:\n=============\n\n"
                        for model_iter_key, data in results_this_essay.items():
                             content_val = data.get('content', 'Error: No content generated.')
                             if not isinstance(content_val, str): content_val = str(content_val) # Ensure string
                             text_content += (
                                  f"=== MODEL: {model_iter_key} ===\n"
                                  f"Vector Store Status (Overall): {'Created' if data.get('vector_store_created', False) else 'Not Created/Failed'}\n"
                                  f"RAG Context Retrieval (This Essay): {'Retrieved' if data.get('rag_success', False) else 'Not Retrieved/Failed/Skipped'}\n"
                                  f"---\n{content_val}\n===\n\n"
                             )
                        essay_files_for_zip["txt"] = (f"{base_filename}_graded.txt", text_content.encode('utf-8'))
                    except Exception as e:
                        st.error(f"Error generating TXT content for {essay_name}: {e}")


                    # 2. PDF file
                    pdf_bytes = save_pdf(essay_name, results_this_essay) # Function handles internal errors
                    if pdf_bytes:
                        essay_files_for_zip["pdf"] = (f"{base_filename}_graded.pdf", pdf_bytes)

                    # 3. CSV file (Comparison Table for this essay)
                    if results_df_this_essay is not None:
                         try:
                              csv_buffer = io.StringIO()
                              results_df_this_essay.to_csv(csv_buffer, index=False, encoding='utf-8')
                              essay_files_for_zip["csv"] = (f"{base_filename}_summary.csv", csv_buffer.getvalue().encode('utf-8'))
                         except Exception as e:
                              st.error(f"Error generating CSV content for {essay_name}: {e}")

                    # Store this essay's generated file data for the bulk zip
                    all_essay_data_for_zip.append(essay_files_for_zip)


                    # --- Download Button for This Essay ---
                    if essay_files_for_zip:
                         try:
                              essay_zip_buffer = io.BytesIO()
                              with zipfile.ZipFile(essay_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                                   for file_type, (filename, file_data) in essay_files_for_zip.items():
                                       # Ensure file_data is bytes
                                       if isinstance(file_data, bytes):
                                           zipf.writestr(filename, file_data)
                                       else:
                                            st.warning(f"Skipping file {filename} in ZIP due to non-bytes data.")

                              essay_zip_buffer.seek(0)

                              st.download_button(
                                   label=f"📥 Download Results for '{essay_name}' (ZIP)",
                                   data=essay_zip_buffer,
                                   file_name=f"{base_filename}_graded_results.zip",
                                   mime="application/zip",
                                   key=f"zip_button_{base_filename}_{i}" # Unique key per essay using index
                              )
                         except Exception as e:
                              st.error(f"Error creating ZIP for {essay_name}: {e}")

                    st.markdown("---") # Separator between essays

                # Update overall progress to 100%
                overall_progress_bar.progress(1.0, text="Finished processing all essays!")

                # --- Final Download Button for All Essays ---
                if len(all_essay_data_for_zip) > 0: # Changed condition to allow download even for 1 essay
                     st.header(" Zipped Results (All Processed Essays)")
                     st.info("Download a single ZIP file containing all generated TXT, PDF (if successful), and CSV files for all processed essays.")
                     try:
                         all_essays_zip_buffer = io.BytesIO()
                         with zipfile.ZipFile(all_essays_zip_buffer, "w", zipfile.ZIP_DEFLATED) as bulk_zipf:
                              for essay_files_data in all_essay_data_for_zip:
                                  # Check if dict is not empty before iterating
                                  if isinstance(essay_files_data, dict):
                                       for file_type, (filename, file_data) in essay_files_data.items():
                                           if isinstance(file_data, bytes): # Ensure data is bytes
                                                bulk_zipf.writestr(filename, file_data) # Use the prepared filenames
                                           else:
                                                st.warning(f"Skipping file {filename} in bulk ZIP due to non-bytes data.")


                         all_essays_zip_buffer.seek(0)
                         st.download_button(
                              label="📦 Download ALL Results (Single ZIP)",
                              data=all_essays_zip_buffer,
                              file_name="ALL_graded_essays_results.zip",
                              mime="application/zip",
                              key="zip_button_all_essays"
                         )
                     except Exception as e:
                         st.error(f"❌ Error creating the combined ZIP file for all essays: {e}")
                         st.error(f"Traceback: {traceback.format_exc()}")


            except ValueError as ve: # Catch specific errors like empty API key
                 st.error(f"❌ Input Error: {ve}")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred during setup or processing: {e}")
                st.error("Traceback:")
                st.code(traceback.format_exc())

# --- End Main Logic ---

if __name__ == "__main__":
    main()
