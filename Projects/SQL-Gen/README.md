# SQL Query Generator with Iterative Auto-Fix

This is a Streamlit application that generates SQL queries from natural language descriptions and automatically fixes SQL errors using iterative techniques.

## Features

- **Natural Language to SQL**: Convert plain English descriptions into DuckDB-compatible SQL queries
- **Iterative Auto-Fix**: Automatically detects and fixes SQL errors using multiple strategies:
  - Rule-based transformations
  - Fuzzy column mapping
  - Numeric type casting
  - AI model rewrites
- **File Upload Support**: Upload CSV, JSON, PDF, or text files for analysis
- **Schema-Aware Generation**: Uses uploaded data schema to generate more accurate SQL
- **DuckDB Integration**: Runs queries locally using DuckDB for fast execution

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables** (optional):
   ```bash
   export GOOGLE_API_KEY="your_google_api_key"
   export OPENAI_API_KEY="your_openai_api_key"  # Optional for embeddings
   ```

## Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

1. **Upload a File** (optional): Upload CSV, JSON, PDF, or text files to provide context for SQL generation
2. **Describe Your Query**: Enter a natural language description of the SQL you want
3. **Generate & Run**: Click "Generate SQL & Run (auto-fix)" to create and execute the query
4. **Review Results**: View the query results and auto-fix log

## Configuration

The application includes several configurable settings in the sidebar:
- **Preview candidate fixes**: Show SQL candidates before execution
- **Max model rewrites**: Number of AI-generated fixes per error
- **Max total attempts**: Maximum attempts before giving up

## Supported File Types

- **CSV**: Automatically parsed into a DataFrame
- **JSON**: Normalized into tabular format
- **PDF**: Text extracted using multiple methods (pdfplumber, PyPDF2, OCR)
- **Text**: Plain text files

## Error Handling

The application uses sophisticated error analysis to automatically fix common SQL issues:
- Column name mismatches
- Function argument errors
- Syntax errors
- Date/time formatting issues
- Type casting problems

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`
- Google API key for AI model access (optional but recommended)

## Troubleshooting

If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

The warnings about "missing ScriptRunContext" are normal when importing the module outside of Streamlit and can be ignored.
