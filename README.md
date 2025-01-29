# XLS Batch Processor

**XLS Batch Processor** is a [Streamlit](https://streamlit.io/) application for **batch processing** spreadsheet data (CSV or Excel) using [OpenAI’s Batch API](https://platform.openai.com/docs/guides/batch).  

## Features

- **Simple UI**: Upload a .csv, .xls, or .xlsx file; choose which columns to feed into an LLM; define output columns; then submit everything for batch processing.  
- **Flexible Instructions**: Provide free‐form text instructions describing how the LLM should transform or analyze each row.  
- **Automatic Chunking**: The app splits your data into smaller “batches” if your request or token count risks exceeding the rate limits for your chosen model.  
- **Multiple Models**: Pick from five models:
  1. `gpt-4o`
  2. `gpt-4o-2024-08-06`
  3. `gpt-4o-mini`
  4. `gpt-4-turbo`
  5. `gpt-4`
- **Progress Display**: Watch the status in real time, including a countdown for the next update.  
- **Output Merging**: The resulting data is merged back into your original file, preserving existing columns and adding new ones.  
- **Session State**: Your data and logs persist in the session, so you can download the final spreadsheet and the API logs **without** losing your place.  
- **Reset Button**: Easily clear the session if you want to upload a new file, while keeping your model choice and API key in the sidebar.

## How It Works

1. **Upload Spreadsheet**  
   - Accepts CSV or Excel files up to the size/memory constraints of Streamlit.  
2. **Enter Instructions**  
   - Example: “Summarize column ‘Description’ in German and provide a ‘Short Title’ in English.”  
3. **Configure Input & Output Columns**  
   - Choose which columns to feed as prompt data.  
   - Specify which columns you want in the final output.  
4. **Submit**  
   - The app converts your data into JSON lines for the Batch API.  
   - Large data is split into multiple chunks to respect rate limits (tokens/min and requests/min).  
   - Each chunk is uploaded, processed, and merged automatically.  
5. **Review & Download**  
   - View the combined output in the app.  
   - Download the final CSV/XLSX and the logs of every API call.

## Setup & Deployment

1. **Local Use**  
   - Clone or download the repo.  
   - In a terminal, run:
     ```bash
     pip install -r requirements.txt
     streamlit run app.py
     ```
   - Open the browser at [http://localhost:8501](http://localhost:8501).

2. **Streamlit Cloud**  
   - Fork or upload this repo to GitHub.  
   - Go to [**share.streamlit.io**](https://share.streamlit.io/) and connect your GitHub.  
   - Deploy the `app.py` file.  
   - Share your unique Streamlit Cloud URL with users.

## Prerequisites

- **Python 3.7+**  
- [**OpenAI API Key**](https://platform.openai.com/)  
- Access to the [**Batch API**](https://platform.openai.com/docs/guides/batch) (some users may need special or beta access).

## Rate Limits

Each chosen model has specific tokens‐per‐minute (TPM) and requests‐per‐minute (RPM) limits. This app automatically chunks data to avoid surpassing them. However, extremely large datasets could still approach daily or monthly limits.

## Security & Disclaimer

- **User Provided Key**: The app prompts for an OpenAI API key, used only client‐side to call the API.  
- **Session Memory**: The data you upload remains in memory while you’re working. For privacy, do not share or host sensitive data unless you trust your environment and that of your cloud provider.  
- **No Guarantee**: This tool is provided “as is” without warranty. Use at your own risk.

## License

[MIT License](https://opensource.org/licenses/MIT) – you can adapt the licensing terms as you see fit.

---

**Enjoy batch-processing your spreadsheet data with GPT models!** If you have any questions or suggestions, feel free to open an issue or a pull request.
