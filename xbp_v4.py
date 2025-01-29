import os
import io
import re
import time
import json
import math
import base64
import pandas as pd
import streamlit as st
import requests

###############################################################################
# Rate-limit info for the 5 allowed models
###############################################################################
MODEL_LIMITS = {
    "gpt-4o":             {"tpm": 2_000_000, "rpm": 10_000},
    "gpt-4o-2024-08-06":  {"tpm": 2_000_000, "rpm": 10_000},
    "gpt-4o-mini":        {"tpm": 10_000_000, "rpm": 10_000},
    "gpt-4-turbo":        {"tpm": 800_000,   "rpm": 10_000},
    "gpt-4":              {"tpm": 300_000,   "rpm": 10_000},
}

###############################################################################
# Utility to parse out triple-backticks or code fences from LLM responses
###############################################################################
def clean_code_fences(text):
    """
    Removes triple backticks (```), and any ```json or ``` lines, so we can parse
    the inner JSON properly.
    """
    # Remove lines with ```json or ``` alone
    text = re.sub(r"```(\w+)?", "", text)
    # Remove any trailing ```
    text = text.replace("```", "")
    return text.strip()


###############################################################################
# The main app
###############################################################################
def main():
    st.title("XLS Batch Processor")

    # --- Sidebar with Model + API key (Always visible) ---
    with st.sidebar:
        st.header("API & Model Settings")

        # Collect the user's OpenAI API key
        # (We do *not* require it just to view the UI.)
        if "openai_api_key" not in st.session_state:
            st.session_state.openai_api_key = ""
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API key:",
            value=st.session_state.openai_api_key,
            type="password"
        )

        # Let user pick from the 5 allowed models
        model_descriptions = {
            "gpt-4o":             "GPT-4o model.",
            "gpt-4o-2024-08-06":  "GPT-4o from 2024-08-06.",
            "gpt-4o-mini":        "GPT-4o mini: smaller/faster variant.",
            "gpt-4-turbo":        "GPT-4 turbo model.",
            "gpt-4":              "Official GPT-4 model."
        }
        if "chosen_model" not in st.session_state:
            st.session_state.chosen_model = "gpt-4o"
        llm_options = list(model_descriptions.keys())
        st.session_state.chosen_model = st.selectbox(
            "Choose a model (LLM) to use:",
            llm_options,
            index=llm_options.index(st.session_state.chosen_model),
            help="Only these 5 are available."
        )
        st.caption(f"**Model info:** {model_descriptions[st.session_state.chosen_model]}")

    # --- "Reset" button to clear everything except API key + model ---
    if st.button("Reset everything except API key and model"):
        for key in list(st.session_state.keys()):
            if key not in ["openai_api_key", "chosen_model"]:
                del st.session_state[key]
        st.experimental_rerun()

    # --- File upload + instructions (always shown) ---
    uploaded_file = st.file_uploader("Upload your CSV/XLS/XLSX here:", type=["csv", "xls", "xlsx"])
    file_format = None
    df = pd.DataFrame()

    if uploaded_file:
        file_name_lower = uploaded_file.name.lower()
        if file_name_lower.endswith(".csv"):
            file_format = "csv"
            df = pd.read_csv(uploaded_file)
        else:
            file_format = "excel"
            df = pd.read_excel(uploaded_file)

    user_instructions = st.text_area(
        "Enter instructions for how the LLM should process each row:",
        height=150,
        help=("Example: Summarize the 'Description' column, generate a tagline, etc.")
    )

    st.subheader("Define Input Structure")
    if not df.empty:
        st.write("Preview of uploaded data (first 5 rows):")
        st.dataframe(df.head(5))
        columns = list(df.columns)
    else:
        columns = []

    input_columns_selected = st.multiselect(
        "Which columns contain relevant input data for the LLM?",
        columns
    )
    column_structure = {}
    for col in input_columns_selected:
        desc = st.text_input(f"Description for '{col}'", value=f"This is {col}")
        column_structure[col] = desc

    st.subheader("Define Output Structure")
    desired_output_cols_text = st.text_area(
        "Enter your desired output columns (comma separated).",
        value="MISMATCH REASON, MISMATCH RATE"
    )
    desired_output_cols = [c.strip() for c in desired_output_cols_text.split(",") if c.strip()]

    # --- If user clicks "Create & Submit" button, we do the entire pipeline ---
    if st.button("Create & Submit Batch to OpenAI"):
        # Check if we have an API key
        if not st.session_state.openai_api_key:
            st.error("Please provide your OpenAI API key before submitting.")
            st.stop()

        # Check if file is uploaded
        if uploaded_file is None or df.empty:
            st.error("Please upload a CSV/XLS/XLSX file before submitting.")
            st.stop()

        # Build JSONL
        jsonl_bytes, preview_requests, total_token_est, total_requests = create_jsonl_for_batch(
            df,
            st.session_state.chosen_model,
            user_instructions,
            column_structure,
            desired_output_cols,
            input_columns_selected
        )

        with st.expander("View Sent Data (JSONL Preview)"):
            lines = jsonl_bytes.decode("utf-8").split("\n")
            st.code("\n".join(lines[:5]) + ("\n..." if len(lines) > 5 else ""), language="json")

        st.write(f"**Approx. Total Tokens:** {total_token_est}")
        st.write(f"**Total Requests (rows):** {total_requests}")

        # Get rate limits for chosen model
        rate_info = MODEL_LIMITS.get(st.session_state.chosen_model, {"tpm": 2_000_000, "rpm": 10_000})
        max_tpm = rate_info["tpm"]
        max_rpm = rate_info["rpm"]

        st.info("Splitting data into smaller parts if needed to avoid rate-limit spikes...")

        # Parse lines from JSONL for chunking
        splitted_lines = jsonl_bytes.decode("utf-8").split("\n")
        lines_data = [json.loads(line) for line in splitted_lines if line.strip()]

        # We'll make multiple "chunks" so no chunk exceeds max_rpm or max_tpm
        chunks = []
        current_chunk = []
        chunk_tokens = 0
        chunk_requests = 0

        for line_obj in lines_data:
            # We only do a fallback ~100 tokens if we didn't store actual estimates
            est_tokens = 100
            if "estimated_tokens" in line_obj.get("body", {}):
                est_tokens = line_obj["body"]["estimated_tokens"]

            if (chunk_requests + 1 > max_rpm) or (chunk_tokens + est_tokens > max_tpm):
                chunks.append(current_chunk)
                current_chunk = [line_obj]
                chunk_tokens = est_tokens
                chunk_requests = 1
            else:
                current_chunk.append(line_obj)
                chunk_tokens += est_tokens
                chunk_requests += 1

        if current_chunk:
            chunks.append(current_chunk)

        total_chunks = len(chunks)
        st.write(f"**Number of parts (batches) to send:** {total_chunks}")

        # We'll track all output lines + a communication log
        all_output_lines = []
        api_communication_log = []

        def log_request_response(label, request_info, response_text):
            api_communication_log.append(f"=== {label} ===")
            if isinstance(request_info, dict):
                api_communication_log.append("Request:")
                api_communication_log.append(json.dumps(request_info, indent=2))
            else:
                api_communication_log.append(f"Request:\n{request_info}")
            api_communication_log.append("Response:")
            api_communication_log.append(response_text)
            api_communication_log.append("\n")

        for idx, chunk_data in enumerate(chunks, start=1):
            st.write(f"### Sending part {idx}/{total_chunks}")
            chunk_jsonl = "\n".join(json.dumps(obj) for obj in chunk_data).encode("utf-8")

            # 1) Upload chunk
            upload_label = f"Chunk {idx} - Upload to /v1/files"
            upload_req_info = f"(Binary JSONL, size={len(chunk_jsonl)} bytes)"
            try:
                upload_resp = requests.post(
                    "https://api.openai.com/v1/files",
                    headers={"Authorization": f"Bearer {st.session_state.openai_api_key}"},
                    files={
                        "file": (f"chunk_{idx}.jsonl", chunk_jsonl, "application/jsonl"),
                        "purpose": (None, "batch")
                    },
                    timeout=60
                )
                log_request_response(upload_label, upload_req_info, upload_resp.text)
            except Exception as e:
                st.error(f"Upload chunk {idx} failed: {e}")
                st.stop()

            if upload_resp.status_code != 200:
                st.error(f"Upload chunk {idx} failed. Code={upload_resp.status_code}.")
                st.stop()

            upload_json = upload_resp.json()
            input_file_id = upload_json["id"]
            st.write(f"Chunk {idx} file uploaded. File ID: {input_file_id}")

            # 2) Create batch
            create_label = f"Chunk {idx} - Create Batch"
            create_req_body = {
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h"
            }
            try:
                create_resp = requests.post(
                    "https://api.openai.com/v1/batches",
                    headers={
                        "Authorization": f"Bearer {st.session_state.openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json=create_req_body,
                    timeout=60
                )
                log_request_response(create_label, create_req_body, create_resp.text)
            except Exception as e:
                st.error(f"Create batch for chunk {idx} failed: {e}")
                st.stop()

            if create_resp.status_code != 200:
                st.error(f"Create batch for chunk {idx} failed. Code={create_resp.status_code}.")
                st.stop()

            batch_json = create_resp.json()
            batch_id = batch_json["id"]
            st.write(f"Chunk {idx} batch created. Batch ID: {batch_id}")

            # 3) Poll with single status line & 20s countdown
            poll_interval = 20
            prev_status = None
            same_status_count = 0
            status_placeholder = st.empty()
            countdown_placeholder = st.empty()

            while True:
                retrieve_label = f"Chunk {idx} - Retrieve Batch {batch_id}"
                try:
                    retrieve_resp = requests.get(
                        f"https://api.openai.com/v1/batches/{batch_id}",
                        headers={"Authorization": f"Bearer {st.session_state.openai_api_key}"},
                        timeout=30
                    )
                    log_request_response(retrieve_label, {}, retrieve_resp.text)
                except Exception as e:
                    status_placeholder.warning(f"Failed to retrieve batch {batch_id} status: {e}")
                    time.sleep(poll_interval)
                    continue

                if retrieve_resp.status_code != 200:
                    status_placeholder.warning(f"Retrieve batch {batch_id} error: {retrieve_resp.text}")
                    time.sleep(poll_interval)
                    continue

                status_resp = retrieve_resp.json()
                status = status_resp["status"]

                if status != prev_status:
                    same_status_count = 1
                    prev_status = status
                    status_placeholder.write(f"Chunk {idx} status: **{status}**")
                else:
                    same_status_count += 1
                    status_placeholder.write(f"Still **{status}** (repeated {same_status_count} times)")

                if status in ["completed", "failed", "cancelled", "expired"]:
                    break

                # countdown for next poll
                for sec in range(poll_interval, 0, -1):
                    countdown_placeholder.info(f"Next status check in {sec} seconds...")
                    time.sleep(1)

            countdown_placeholder.empty()

            if status != "completed":
                st.error(f"Chunk {idx} ended with status: {status}")
                st.stop()

            # 4) Retrieve output
            output_file_id = status_resp["output_file_id"]
            status_placeholder.success(f"Chunk {idx} completed. Output file ID: {output_file_id}")

            retrieve_file_label = f"Chunk {idx} - Retrieve File {output_file_id}"
            try:
                file_resp = requests.get(
                    f"https://api.openai.com/v1/files/{output_file_id}/content",
                    headers={"Authorization": f"Bearer {st.session_state.openai_api_key}"},
                    timeout=60
                )
                log_request_response(retrieve_file_label, {}, file_resp.text)
            except Exception as e:
                st.error(f"Could not retrieve chunk {idx} output content: {e}")
                st.stop()

            if file_resp.status_code != 200:
                st.error(f"Retrieve file content chunk {idx} failed. Code={file_resp.status_code}")
                st.stop()

            chunk_output_content = file_resp.text
            chunk_output_lines = chunk_output_content.strip().split("\n")
            all_output_lines.extend(chunk_output_lines)

            # If more chunks remain, wait 75s
            if idx < total_chunks:
                st.write("Waiting 75s before sending the next chunk...")
                for i in range(75, 0, -1):
                    st.write(f"Next chunk in {i} seconds...")
                    time.sleep(1)

        # All chunks done
        with st.expander("View Received Data (JSONL Output Preview)"):
            st.code("\n".join(all_output_lines[:5]) + ("\n..." if len(all_output_lines) > 5 else ""), language="json")

        final_output_content = "\n".join(all_output_lines)
        result_df = merge_batch_responses(df, final_output_content, desired_output_cols)
        st.subheader("Merged Data Preview")
        st.dataframe(result_df.head(5))

        # Save the final DataFrame + logs in session state (so no reset on downloads)
        st.session_state["final_df"] = result_df
        st.session_state["final_logs"] = "\n".join(api_communication_log)
        st.session_state["final_file_format"] = file_format

    # --- Offer download buttons if we have final results in session state ---
    if "final_df" in st.session_state and not st.session_state["final_df"].empty:
        st.write("## Download Results")
        result_df = st.session_state["final_df"]
        file_format = st.session_state["final_file_format"]
        # Prepare data
        if file_format == "csv":
            file_data = result_df.to_csv(index=False).encode("utf-8")
            file_name = "processed_output.csv"
            mime_type = "text/csv"
        else:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Sheet1")
            file_data = excel_buffer.getvalue()
            file_name = "processed_output.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        st.download_button(
            label="Download Processed Output",
            data=file_data,
            file_name=file_name,
            mime=mime_type
        )

    if "final_logs" in st.session_state:
        st.write("## Download API Communication Logs")
        api_log_str = st.session_state["final_logs"]
        st.download_button(
            label="Download API Communications (txt)",
            data=api_log_str.encode("utf-8"),
            file_name="api_communications_log.txt",
            mime="text/plain"
        )
        with st.expander("Detailed API Communication Logs"):
            st.text(api_log_str)


###############################################################################
# Helper functions
###############################################################################

def approximate_token_count(sys_msg, user_msg):
    """
    Very rough token estimation for system+user messages.
    """
    total_chars = len(sys_msg) + len(user_msg)
    return math.ceil(total_chars / 4)

def create_jsonl_for_batch(
    df,
    model_name,
    user_instructions,
    column_structure,
    desired_output_cols,
    input_columns_selected
):
    """
    Return (jsonl_bytes, preview_requests, total_token_est, total_requests).
    Each line is a JSON with {custom_id, method, url, body{model, messages...}}.
    """
    json_lines = []
    preview_requests = []
    total_token_est = 0
    total_requests = 0

    for idx, row in df.iterrows():
        system_message = (
            "You are a helpful data-processing assistant. "
            "You will receive row data (with column descriptions), plus instructions. "
            "You MUST return results in JSON that strictly contains these keys:\n"
            f"{desired_output_cols}\n"
            "No additional commentary."
        )

        user_message_parts = ["### COLUMN DATA:"]
        for col in input_columns_selected:
            col_desc = column_structure.get(col, "No description provided")
            col_value = row[col]
            user_message_parts.append(f"- {col} ({col_desc}): {col_value}")

        user_message_parts.append("### INSTRUCTIONS:")
        user_message_parts.append(user_instructions)
        user_message_parts.append("### REQUIRED OUTPUT KEYS:")
        user_message_parts.append(str(desired_output_cols))

        user_full_message = "\n".join(user_message_parts)

        # approximate tokens
        est_tokens = approximate_token_count(system_message, user_full_message)
        total_token_est += est_tokens
        total_requests += 1

        request_body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_full_message},
            ],
            "temperature": 0.0,
            "max_tokens": 500,
            # optionally store an "estimated_tokens" if you want to read it in chunking
            # "estimated_tokens": est_tokens
        }

        batch_request = {
            "custom_id": f"row-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request_body
        }
        if idx < 3:
            preview_requests.append(batch_request)

        json_lines.append(json.dumps(batch_request))

    jsonl_data = "\n".join(json_lines).encode("utf-8")
    return jsonl_data, preview_requests, total_token_est, total_requests


def merge_batch_responses(original_df, output_content, desired_output_cols):
    """
    The Batch API output is JSONL. Each line has 'custom_id' and 'response' or 'error'.
    We'll parse it, match back to the row, and store the final columns.
    Also handle triple-backtick code fences if present.
    """
    result_df = original_df.copy()
    lines = output_content.strip().split("\n")
    row_outputs = {}

    for line in lines:
        data = json.loads(line)
        custom_id = data.get("custom_id", "")
        # If there's an error object, store that in each desired col
        error_info = data.get("error")
        if error_info:
            row_index = parse_custom_id(custom_id)
            row_outputs[row_index] = {
                col: f"ERROR: {error_info}" for col in desired_output_cols
            }
            continue

        # Normal response path
        response_body = data.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if not choices:
            continue
        content = choices[0]["message"]["content"]
        # remove any triple-backtick code fences so we can parse real JSON
        cleaned_content = clean_code_fences(content)

        try:
            content_json = json.loads(cleaned_content)
        except:
            content_json = {"_raw_output": cleaned_content}

        row_index = parse_custom_id(custom_id)
        out_dict = {}
        for col in desired_output_cols:
            if col in content_json:
                out_dict[col] = content_json[col]
            else:
                out_dict[col] = content_json.get("_raw_output", "N/A")
        row_outputs[row_index] = out_dict

    # Add new columns if missing
    for col in desired_output_cols:
        if col not in result_df.columns:
            result_df[col] = None

    # Place results into the DataFrame
    for idx, vals in row_outputs.items():
        for col in desired_output_cols:
            result_df.at[idx, col] = vals[col]

    return result_df

def clean_code_fences(text):
    """
    Removes triple backticks (```), or ```json etc. lines, so we can parse inner JSON.
    """
    # Remove lines with ```json or ``` alone
    text = re.sub(r"```(\w+)?", "", text)
    # Remove any leftover ```
    text = text.replace("```", "")
    return text.strip()

def parse_custom_id(custom_id):
    if custom_id.startswith("row-"):
        try:
            return int(custom_id.split("row-")[1])
        except:
            return None
    return None


# Run the app
if __name__ == "__main__":
    main()
