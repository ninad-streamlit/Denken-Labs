# app.py — Iterative error-analyzer + auto-fixer (DuckDB-focused)
import os
import re
import io
import json
import textwrap
import math
import difflib
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

import streamlit as st
import pandas as pd
import google.generativeai as genai  # type: ignore

# Optional PDF/OCR libs (import dynamically)
try:
    import pdfplumber  # type: ignore[import-not-found]
except Exception:
    pdfplumber = None

try:
    from PyPDF2 import PdfReader  # type: ignore[import-not-found]
except Exception:
    PdfReader = None

try:
    from pdf2image import convert_from_bytes  # type: ignore[import-not-found]
except Exception:
    convert_from_bytes = None

try:
    import pytesseract  # type: ignore[import-not-found]
except Exception:
    pytesseract = None

# Optional connectors/engines
try:
    import databricks.sql as dbsql  # type: ignore[import-not-found]
except Exception:
    dbsql = None

try:
    import duckdb  # type: ignore[import-not-found]
except Exception:
    duckdb = None

try:
    import openai  # type: ignore[import-not-found]
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    if OPENAI_AVAILABLE:
        openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception:
    openai = None
    OPENAI_AVAILABLE = False

# ---------------------- CONFIG ----------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or "AIzaSyBC1CsvuS3_3sX7Gadcijo8kVYaMUYNAXM"
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.5-flash")

DB_HOST = os.getenv("DATABRICKS_HOST", "")
DB_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")
DB_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

# Model setup (guarded)
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)  # type: ignore
    except Exception:
        pass

try:
    model = genai.GenerativeModel(MODEL_NAME)  # type: ignore[reportPrivateImportUsage]
except Exception:
    model = None

# Tuning
MAX_MODEL_REWRITES = 3          # how many distinct rewrites to request from model
MAX_TOTAL_ATTEMPTS = 12        # max candidate attempts before giving up
PREVIEW_BEFORE_APPLY = False   # if True, show candidate SQLs and ask user to confirm before executing

# ---------------------- UTIL HELPERS ----------------------
def safe_get_uploaded_bytes() -> bytes:
    b = st.session_state.get("uploaded_bytes")
    return b if isinstance(b, (bytes, bytearray)) else b""

def extract_text_from_pdf_bytes(raw: bytes) -> str:
    if not raw:
        return ""
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n".join(pages)
                if text.strip():
                    return text
        except Exception:
            pass
    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            text = "\n".join(pages)
            if text.strip():
                return text
        except Exception:
            pass
    if convert_from_bytes is not None and pytesseract is not None:
        try:
            images = convert_from_bytes(raw)
            pages = []
            for img in images:
                try:
                    pages.append(pytesseract.image_to_string(img))
                except Exception:
                    pages.append("")
            text = "\n".join(pages)
            if text.strip():
                return text
        except Exception:
            pass
    return ""

def extract_text_from_file(uploaded_file) -> Tuple[str, Optional[pd.DataFrame]]:
    fname = getattr(uploaded_file, "name", "").lower()
    try:
        raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    except Exception:
        try:
            uploaded_file.seek(0)
            raw = uploaded_file.read()
        except Exception:
            raw = b""

    if not isinstance(raw, (bytes, bytearray)):
        raw = (raw or "").encode("utf-8", errors="replace")

    if fname.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(raw))
            return df.to_csv(index=False), df
        except Exception:
            try:
                return raw.decode("utf-8", errors="replace"), None
            except Exception:
                return "", None

    if fname.endswith(".json"):
        try:
            obj = json.loads(raw.decode("utf-8", errors="replace"))
            txt = json.dumps(obj, indent=2, ensure_ascii=False)
            try:
                df = pd.json_normalize(obj)
            except Exception:
                df = None
            return txt, df
        except Exception:
            return raw.decode("utf-8", errors="replace"), None

    if fname.endswith(".txt"):
        try:
            return raw.decode("utf-8", errors="replace"), None
        except Exception:
            return "", None

    if fname.endswith(".pdf"):
        text = extract_text_from_pdf_bytes(raw)
        return text, None

    try:
        return raw.decode("utf-8", errors="replace"), None
    except Exception:
        return "", None

def strip_code_fences(s: str) -> str:
    if not s:
        return ""
    s2 = s.replace("```sql", "```")
    while "```" in s2:
        s2 = s2.replace("```", "")
    return s2.strip()

def add_limit_if_missing(sql: str, limit: int = 1000) -> str:
    if re.search(r"\blimit\b", sql, flags=re.I):
        return sql
    sql = sql.rstrip().rstrip(";")
    return f"{sql}\nLIMIT {limit}"

def is_safe_select(sql: str) -> bool:
    if not isinstance(sql, str):
        return False
    s = sql.strip().lower()
    if not s:
        return False
    if not (s.startswith("select") or s.startswith("describe") or s.startswith("show")):
        return False
    if ";" in s.replace(";", "", 1):
        return False
    banned = ["insert ", "update ", "delete ", "drop ", "truncate ", "create ", "alter ", "merge ", "replace "]
    for b in banned:
        if b in s:
            return False
    return True

def run_databricks_query(sql: str, server_hostname: str, http_path: str, access_token: str, timeout_seconds: int = 60) -> pd.DataFrame:
    if dbsql is None:
        raise RuntimeError("databricks-sql-connector not installed. Install with: pip install databricks-sql-connector[pyarrow]")
    s = sql.strip().lower()
    if not (s.startswith("select") or s.startswith("describe") or s.startswith("show")):
        raise ValueError("Only read-only queries (SELECT / DESCRIBE / SHOW) are allowed via this helper.")
    with dbsql.connect(server_hostname=server_hostname, http_path=http_path, access_token=access_token, timeout=timeout_seconds) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols)
        return df

# ---------------------- Embeddings utilities (optional) ----------------------
def openai_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    if not OPENAI_AVAILABLE or openai is None:
        return None
    try:
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        resp = openai.Embedding.create(input=texts, model=model_name)
        return [e["embedding"] for e in resp["data"]]
    except Exception:
        return None

def l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))

def cosine_sim(a: List[float], b: List[float]) -> float:
    an = l2_norm(a)
    bn = l2_norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (an * bn)

# ---------------------- Schema-aware SQL generation ----------------------
def build_schema_text(df: Optional[pd.DataFrame], max_cols: int = 12, max_rows: int = 5) -> str:
    if df is None or df.shape[1] == 0:
        return ""
    cols = list(df.columns)[:max_cols]
    col_desc = "\n".join(f"- {c} ({str(df[c].dtype)})" for c in cols)
    sample = df[cols].head(max_rows).to_dict(orient="records")
    sample_text = json.dumps(sample, ensure_ascii=False, indent=2)
    return f"Columns:\n{col_desc}\n\nSample rows:\n{sample_text}\n"

def generate_sql_with_schema(nl_instruction: str, df: Optional[pd.DataFrame]) -> str:
    """Generate SQL while including a short schema+samples in the prompt. Guard model availability."""
    if model is None:
        raise RuntimeError("Model not available. Configure GOOGLE_API_KEY and MODEL_NAME.")
    schema_text = build_schema_text(df) or "(no schema available)"
    template = textwrap.dedent(
        """
        You are given a user's natural-language request and a short schema + sample rows.
        Produce a single DuckDB-compatible SQL statement ONLY (no explanation). Use one of these table names:
        - uploaded_table (the CSV/JSON table columns are shown below)
        - document(paragraph_id, text) (for free text)

        Schema and samples:
        {schema}

        Instruction:
        {instruction}

        Notes for the SQL:
        - Prefer exact column names from the schema. If the instruction mentions a concept that matches multiple columns, choose the most likely column.
        - If a numeric comparison is required but the column is textual (e.g. contains currency symbols), return an expression that converts it to numeric, using REGEXP_REPLACE or CAST, e.g. CAST(REGEXP_REPLACE(balance_eur, '[^0-9.-]','') AS DOUBLE) > 100
        - Use DuckDB-compatible syntax (avoid INTERVAL 'x' DAY; prefer date arithmetic or NOW() - x * INTERVAL).
        - Return only the final SQL statement.
        """
    ).strip()
    prompt = template.format(schema=schema_text, instruction=nl_instruction)
    # safe call to model
    resp = model.generate_content(prompt)
    sql_raw = getattr(resp, "text", "") or ""
    return strip_code_fences(sql_raw)

# ---------------------- semantic / fuzzy mapping & cast injection ----------------------
def fuzzy_map_columns(sql: str, df: pd.DataFrame, cutoff: float = 0.65) -> Tuple[str, Dict[str, str]]:
    if df is None or df.shape[1] == 0:
        return sql, {}
    cols = list(df.columns)
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sql))
    keywords = {
        "select","from","where","and","or","group","by","order","limit","as","count","sum","avg",
        "min","max","join","on","left","right","inner","outer","desc","asc","having","distinct",
        "with","union","all","case","when","then","else","end","is","null","not","in","between","like"
    }
    candidates = [t for t in tokens if t.lower() not in keywords and not t.isnumeric()]
    replacements: Dict[str, str] = {}
    new_sql = sql

    # Try embeddings-based mapping first (if available)
    try:
        if OPENAI_AVAILABLE and openai is not None and candidates:
            emb_texts = []
            for c in cols:
                try:
                    sample_vals = [str(v) for v in df[c].dropna().astype(str).head(3).tolist()]
                except Exception:
                    sample_vals = []
                emb_texts.append(c + " " + " | ".join(sample_vals))
            col_embs = openai_embeddings(emb_texts)
            cand_embs = openai_embeddings(candidates)
            if col_embs is not None and cand_embs is not None:
                for i, cand in enumerate(candidates):
                    best_idx = None
                    best_score = -1.0
                    for j, ce in enumerate(col_embs):
                        score = cosine_sim(cand_embs[i], ce)
                        if score > best_score:
                            best_score = score
                            best_idx = j
                    if best_score >= cutoff and best_idx is not None:
                        replacements[cand] = cols[best_idx]
                for orig, mapped in replacements.items():
                    new_sql = re.sub(r"\b" + re.escape(orig) + r"\b", mapped, new_sql)
                if replacements:
                    return new_sql, replacements
    except Exception:
        replacements = {}

    # Fallback to difflib
    for cand in candidates:
        if cand in cols:
            continue
        matches = difflib.get_close_matches(cand, cols, n=1, cutoff=cutoff)
        if matches:
            replacements[cand] = matches[0]
    for orig, mapped in replacements.items():
        new_sql = re.sub(r"\b" + re.escape(orig) + r"\b", mapped, new_sql)
    return new_sql, replacements

def inject_numeric_casts(sql: str, df: pd.DataFrame) -> str:
    def repl(m):
        col = m.group("col")
        op = m.group("op")
        val = m.group("val")
        if col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    return f"{col} {op} {val}"
                else:
                    expr = f"CAST(REGEXP_REPLACE({col}, '[^0-9.-]', '') AS DOUBLE)"
                    return f"{expr} {op} {val}"
            except Exception:
                return m.group(0)
        return m.group(0)
    pattern = re.compile(r"(?P<col>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>>=|<=|>|<|=)\s*(?P<val>-?\d+(\.\d+)?)")
    return pattern.sub(repl, sql)

# ---------------------- Auto-fix logging helpers ----------
def _init_auto_fix_log():
    if "auto_fix_log" not in st.session_state:
        st.session_state["auto_fix_log"] = []

def log_auto_fix(entry: dict):
    _init_auto_fix_log()
    entry = dict(entry)
    entry.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    st.session_state["auto_fix_log"].append(entry)

# ---------------------- Error analysis + candidate generation ----------
ERROR_PATTERNS = [
    ("no_such_column", re.compile(r"(no such column|unknown column|column .* not found|binding .* not found|referenced column)", re.I)),
    ("function_args", re.compile(r"(wrong number of arguments|argument mismatch|expected .* arguments|supplied .* arguments)", re.I)),
    ("syntax_error", re.compile(r"(syntax error|parser error|parse error|near \"(.{1,30})\")", re.I)),
    ("date_error", re.compile(r"(date|datetime|interval|time zone|timestamp).*error", re.I)),
    ("table_not_found", re.compile(r"(no such table|table .* does not exist|relation .* does not exist)", re.I)),
]

def classify_error_message(err: str) -> Tuple[str, Optional[re.Match]]:
    """Return a short error type and any regex match for details."""
    for t, pat in ERROR_PATTERNS:
        m = pat.search(err)
        if m:
            return t, m
    return "unknown", None

def rule_based_candidates(sql: str, err_type: str, err_msg: str) -> List[Tuple[str,str]]:
    """
    Returns list of (candidate_sql, note) applying quick deterministic transforms.
    Note: candidate_sql must be full SQL string.
    """
    candidates: List[Tuple[str,str]] = []
    s = sql

    # common function name translations
    func_map = {
        r"\bSTRFTIME\s*\(": "strftime(",
        r"\bTO_CHAR\s*\(": "strftime(",
        r"\bDATE_SUB\s*\(": "",  # leave for model/diff approaches
    }

    # Try replacing INTERVAL '90' DAY style with DuckDB-friendly arithmetic
    if "interval" in s.lower():
        cand = re.sub(r"INTERVAL\s*'(\d+)'\s*DAY", r"(\1) * 1", s, flags=re.I)  # placeholder (model will refine)
        candidates.append((cand, "replace INTERVAL with numeric placeholder"))

    # Replace common 3-arg DATE functions or CAST mismatches by removing bad CASTs
    cand2 = re.sub(r"CAST\(([^)]+) AS DATETIME\)", r"\1", s, flags=re.I)
    if cand2 != s:
        candidates.append((cand2, "remove CAST AS DATETIME"))

    # If user used CURRENT_TIMESTAMP - INTERVAL '90' DAY, convert to date arithmetic (DuckDB)
    if re.search(r"CURRENT_TIMESTAMP\s*-\s*INTERVAL", s, flags=re.I):
        cand3 = re.sub(r"CURRENT_TIMESTAMP\s*-\s*INTERVAL\s*'(\d+)'\s*DAY", r"CURRENT_TIMESTAMP - (\1) * INTERVAL '1 day'", s, flags=re.I)
        candidates.append((cand3, "normalize CURRENT_TIMESTAMP - INTERVAL"))

    # If STRFTIME found but with CAST wrapper, try simpler form
    if re.search(r"STRFTIME\s*\(", s, flags=re.I):
        cand4 = s  # keep as-is, also try removing CASTs
        cand4b = re.sub(r"CAST\(([^)]+) AS DATETIME\)", r"\1", s, flags=re.I)
        if cand4b != s:
            candidates.append((cand4b, "strftime: removed CAST wrappers"))

    # A generic fallback: strip trailing semicolons / extraneous characters often seen in model outputs
    cand_strip = re.sub(r";\s*$", "", s, flags=re.I)
    if cand_strip != s:
        candidates.append((cand_strip, "strip trailing semicolon"))

    # Remove any surrounding prose (if model accidentally included it)
    if "select" not in s.lower().split():
        # no-op here; model rewrite will handle this
        pass

    # uniq
    out = []
    seen = set()
    for c, note in candidates:
        if c not in seen:
            out.append((c, note))
            seen.add(c)
    return out

def generate_model_fixes(sql: str, err_msg: str, df: Optional[pd.DataFrame], n: int = 3) -> List[Tuple[str,str]]:
    """
    Ask the model to propose up to n corrected DuckDB-compatible SQL statements.
    Return list of (sql, note).
    """
    if model is None:
        return []
    schema_text = build_schema_text(df)
    prompt = textwrap.dedent(f"""
        The following SQL produced a parser/runtime error when run in DuckDB.
        Error message:
        {err_msg}

        Schema (table 'uploaded_table' if present):
        {schema_text}

        Rewrite the SQL into up to {n} distinct DuckDB-compatible SQL statements that implement the same intent.
        Return ONLY the SQL statements, separated by a line with exactly three hyphens: ---
        Do NOT include any explanation. Use functions compatible with DuckDB.
    """).strip()
    try:
        resp = model.generate_content(prompt)
        txt = getattr(resp, "text", "") or ""
        txt = strip_code_fences(txt)
        # Split on lines with --- to get multiple candidates
        parts = [p.strip() for p in re.split(r"\n-{3,}\n", txt) if p.strip()]
        out = []
        for i, p in enumerate(parts[:n]):
            out.append((p, f"model_rewrite_candidate_{i+1}"))
        return out
    except Exception:
        return []

# ---------------------- Execution-guided wrapper (with iterative auto-fix) ----------
def try_execute_candidate(con, candidate_sql: str, df: Optional[pd.DataFrame]):
    """
    Try executing candidate_sql in the provided duckdb connection.
    Returns (success_bool, df_or_error)
    """
    try:
        # ensure table registered if df passed
        if df is not None:
            try:
                # First try to unregister if it exists to avoid conflicts
                try:
                    con.unregister("uploaded_table")
                except Exception:
                    pass  # ignore if not registered
                # Register the table
                con.register("uploaded_table", df)
            except Exception as e:
                # If registration fails, log the error but continue
                print(f"Warning: Failed to register uploaded_table: {e}")
                pass
        
        df_res = con.execute(add_limit_if_missing(candidate_sql, 1000)).fetchdf()
        return True, df_res
    except Exception as e:
        return False, str(e)

def generate_and_run_sql(nl_request: str, df: Optional[pd.DataFrame], con) -> pd.DataFrame:
    """
    Top-level wrapper: generate SQL, attempt execution, and auto-fix iteratively by analyzing errors.
    """
    if model is None:
        raise RuntimeError("Model not available - configure GOOGLE_API_KEY and MODEL_NAME.")
    # 1) Generate initial SQL
    sql = generate_sql_with_schema(nl_request, df).strip()
    if not sql:
        raise RuntimeError("Model returned empty SQL")
    st.session_state["generated_sql"] = sql

    attempt_count = 0
    tried_sqls = set()
    candidate_queue: List[Tuple[str,str]] = [(sql, "initial_generated")]

    # Keep looping until we either succeed or exhaust attempts
    while candidate_queue and attempt_count < MAX_TOTAL_ATTEMPTS:
        candidate_sql, reason = candidate_queue.pop(0)
        candidate_sql = candidate_sql.strip()
        if not candidate_sql or candidate_sql in tried_sqls:
            continue
        attempt_count += 1
        tried_sqls.add(candidate_sql)

        # Optionally show preview before applying
        if PREVIEW_BEFORE_APPLY:
            st.info(f"Previewing candidate #{attempt_count}: {reason}")
            st.code(candidate_sql, language="sql")
            if not st.button(f"Run candidate #{attempt_count} now"):
                # if user doesn't press, skip (in interactive mode this blocks; default False is fine)
                pass

        success, result = try_execute_candidate(con, candidate_sql, df)
        if success:
            log_auto_fix({
                "type": "success",
                "attempt": attempt_count,
                "reason": reason,
                "original_sql": sql,
                "fixed_sql": candidate_sql,
                "note": "executed successfully"
            })
            return result  # success

        # If execution failed, analyze error and produce new candidates
        err_msg = str(result)
        err_type, match = classify_error_message(err_msg)

        log_auto_fix({
            "type": "attempt_failed",
            "attempt": attempt_count,
            "reason": reason,
            "error_class": err_type,
            "error_message": err_msg,
            "candidate_sql": candidate_sql
        })

        # 1) quick rule-based candidates
        r_cands = rule_based_candidates(candidate_sql, err_type, err_msg)
        for c_sql, note in r_cands:
            if c_sql and c_sql not in tried_sqls:
                candidate_queue.append((c_sql, f"rule_fix: {note}"))

        # 2) fuzzy column remapping + numeric casts (only if 'no_such_column' or binding)
        if err_type in ("no_such_column", "unknown"):
            try:
                mapped_sql, replacements = fuzzy_map_columns(candidate_sql, df if df is not None else pd.DataFrame())
                if replacements:
                    mapped_sql_casted = inject_numeric_casts(mapped_sql, df if df is not None else pd.DataFrame())
                    if mapped_sql_casted not in tried_sqls:
                        candidate_queue.append((mapped_sql_casted, f"fuzzy_map: {replacements}"))
                        log_auto_fix({
                            "type": "fuzzy_map_generated",
                            "attempt": attempt_count,
                            "replacements": replacements,
                            "candidate_sql": mapped_sql_casted
                        })
            except Exception:
                pass

        # 3) model-based rewrites (ask for up to MAX_MODEL_REWRITES candidates)
        if model is not None and attempt_count < MAX_TOTAL_ATTEMPTS:
            try:
                model_cands = generate_model_fixes(candidate_sql, err_msg, df, n=MAX_MODEL_REWRITES)
                for m_sql, note in model_cands:
                    if m_sql and m_sql not in tried_sqls:
                        candidate_queue.append((m_sql, f"model_rewrite: {note}"))
                        log_auto_fix({
                            "type": "model_candidate_generated",
                            "attempt": attempt_count,
                            "note": note,
                            "candidate_sql": m_sql
                        })
            except Exception:
                pass

    # if exhausted
    raise RuntimeError(f"All automatic repair attempts failed after {attempt_count} tries. Last error: {err_msg}")

# ---------------------- UI ----------------------
st.set_page_config(page_title="SQL-first → iterative auto-fix", layout="wide")
st.title("SQL Query Generator — Iterative Auto-Fix (DuckDB)")

uploaded = st.file_uploader("Upload a source file (optional). If present, the app will run SQL locally.", type=["pdf", "csv", "json", "txt"])

# Persist uploaded bytes safely
if uploaded is not None:
    if st.session_state.get("uploaded_name") != uploaded.name or "uploaded_bytes" not in st.session_state:
        try:
            ub = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        except Exception:
            try:
                uploaded.seek(0)
                ub = uploaded.read()
            except Exception:
                ub = b""
        st.session_state["uploaded_bytes"] = ub if isinstance(ub, (bytes, bytearray)) else b""
        st.session_state["uploaded_name"] = uploaded.name
else:
    # clear uploaded data if user removed file
    st.session_state.pop("uploaded_bytes", None)
    st.session_state.pop("uploaded_name", None)
    st.session_state.pop("extracted_text", None)
    st.session_state.pop("extracted_df", None)

# Auto-extract (lightweight)
extracted_text = st.session_state.get("extracted_text", "")
extracted_df = st.session_state.get("extracted_df", None)

if uploaded is not None and not extracted_text and extracted_df is None:
    ub = safe_get_uploaded_bytes()
    if ub:
        try:
            df_tmp = pd.read_csv(io.BytesIO(ub))
            extracted_df = df_tmp
            extracted_text = extracted_df.to_csv(index=False)
            st.session_state["extracted_df"] = extracted_df
            st.session_state["extracted_text"] = extracted_text
        except Exception:
            try:
                obj = json.loads(ub.decode("utf-8", errors="replace"))
                try:
                    extracted_df = pd.json_normalize(obj)
                except Exception:
                    extracted_df = None
                extracted_text = json.dumps(obj, indent=2, ensure_ascii=False)
                st.session_state["extracted_df"] = extracted_df
                st.session_state["extracted_text"] = extracted_text
            except Exception:
                try:
                    extracted_text = ub.decode("utf-8", errors="replace")
                    st.session_state["extracted_text"] = extracted_text
                except Exception:
                    extracted_text = extract_text_from_pdf_bytes(ub)
                    st.session_state["extracted_text"] = extracted_text

# Preview
if uploaded is not None:
    st.markdown(f"**Uploaded file:** {uploaded.name}")
    if extracted_df is not None:
        st.markdown("**Parsed table (CSV/JSON):**")
        st.dataframe(extracted_df.head(50))
    else:
        with st.expander("Preview extracted text (first 10k chars)"):
            st.text((extracted_text or "(no text extracted)")[:10000])

# SQL-first generation UI
st.markdown("---")
st.markdown("## 1) Generate SQL from natural-language (SQL-first)")
query_prompt = st.text_area("Describe the SQL you want (plain English):", height=140)
gen_button = st.button("Generate SQL & Run (auto-fix)")

# Preview settings
st.sidebar.markdown("### Auto-fix settings")
st.sidebar.checkbox("Preview candidate fixes before executing (slows workflow)", value=PREVIEW_BEFORE_APPLY, key="preview_fixes")
st.sidebar.number_input("Max model rewrites per error", min_value=1, max_value=10, value=MAX_MODEL_REWRITES, key="max_model_rewrites")
st.sidebar.number_input("Max total attempts", min_value=3, max_value=50, value=MAX_TOTAL_ATTEMPTS, key="max_total_attempts")

if gen_button:
    if not query_prompt.strip():
        st.warning("Please enter a description first.")
    else:
        if model is None:
            st.error("Google generative model not available. Configure GOOGLE_API_KEY and MODEL_NAME.")
            st.session_state["generated_sql"] = ""
        else:
            # update run-time settings from sidebar
            PREVIEW_BEFORE_APPLY = st.session_state.get("preview_fixes", PREVIEW_BEFORE_APPLY)
            MAX_MODEL_REWRITES = int(st.session_state.get("max_model_rewrites", MAX_MODEL_REWRITES))
            MAX_TOTAL_ATTEMPTS = int(st.session_state.get("max_total_attempts", MAX_TOTAL_ATTEMPTS))

            try:
                with st.spinner("Generating schema-aware SQL and executing with auto-fix..."):
                    # Prepare duckdb
                    if duckdb is None:
                        st.error("Local SQL execution requires duckdb. Install with: pip install duckdb")
                    else:
                        con = duckdb.connect(database=":memory:")
                        df_for_run = None
                        # register extracted table(s)
                        if extracted_df is not None:
                            df_for_run = extracted_df
                            try:
                                # Unregister first to avoid conflicts
                                try:
                                    con.unregister("uploaded_table")
                                except Exception:
                                    pass
                                con.register("uploaded_table", extracted_df)
                                st.info(f"✅ Registered uploaded_table with {len(extracted_df)} rows and {len(extracted_df.columns)} columns")
                            except Exception as e:
                                st.warning(f"⚠️ Failed to register extracted DataFrame: {e}")
                        else:
                            ub = safe_get_uploaded_bytes()
                            if ub:
                                try:
                                    # Try multiple CSV parsing strategies
                                    tmp_df = None
                                    csv_error_details = ""
                                    
                                    # Strategy 1: Standard pandas CSV parsing
                                    try:
                                        tmp_df = pd.read_csv(io.BytesIO(ub))
                                        st.info("✅ CSV parsed successfully with standard settings")
                                    except Exception as e1:
                                        csv_error_details += f"Standard parsing failed: {str(e1)}\n"
                                        
                                        # Strategy 2: Try with different separators
                                        try:
                                            tmp_df = pd.read_csv(io.BytesIO(ub), sep=None, engine='python')
                                            st.info("✅ CSV parsed successfully with auto-detected separator")
                                        except Exception as e2:
                                            csv_error_details += f"Auto-separator parsing failed: {str(e2)}\n"
                                            
                                            # Strategy 3: Try with error handling
                                            try:
                                                tmp_df = pd.read_csv(io.BytesIO(ub), on_bad_lines='skip', engine='python')
                                                st.warning("⚠️ CSV parsed with some lines skipped due to formatting issues")
                                            except Exception as e3:
                                                csv_error_details += f"Error-handling parsing failed: {str(e3)}\n"
                                                
                                                # Strategy 4: Try with quoting
                                                try:
                                                    tmp_df = pd.read_csv(io.BytesIO(ub), quoting=1, engine='python')  # QUOTE_ALL
                                                    st.info("✅ CSV parsed successfully with quoting enabled")
                                                except Exception as e4:
                                                    csv_error_details += f"Quoted parsing failed: {str(e4)}\n"
                                                    
                                                    # Strategy 5: Manual line-by-line parsing as last resort
                                                    try:
                                                        lines = ub.decode('utf-8', errors='replace').split('\n')
                                                        st.warning("⚠️ Attempting manual CSV parsing...")
                                                        
                                                        # Find the line with the error (line 13 based on error message)
                                                        if len(lines) >= 13:
                                                            st.error(f"Problematic line 13: {repr(lines[12])}")
                                                            
                                                        # Try to fix common issues
                                                        fixed_lines = []
                                                        for i, line in enumerate(lines):
                                                            if i == 12:  # Line 13 (0-indexed)
                                                                # Try to fix common CSV issues
                                                                if line.count(',') == 1 and not (line.startswith('"') and line.endswith('"')):
                                                                    # Wrap the entire line in quotes if it has one comma but isn't quoted
                                                                    line = f'"{line}"'
                                                                    st.info(f"Fixed line 13 by adding quotes: {line}")
                                                            fixed_lines.append(line)
                                                        
                                                        # Try parsing the fixed content
                                                        fixed_content = '\n'.join(fixed_lines)
                                                        tmp_df = pd.read_csv(io.StringIO(fixed_content))
                                                        st.success("✅ CSV parsed successfully after manual fixes")
                                                    except Exception as e5:
                                                        csv_error_details += f"Manual parsing failed: {str(e5)}\n"
                                                        raise Exception(f"All CSV parsing strategies failed:\n{csv_error_details}")
                                    
                                    if tmp_df is not None:
                                        df_for_run = tmp_df
                                        # Unregister first to avoid conflicts
                                        try:
                                            con.unregister("uploaded_table")
                                        except Exception:
                                            pass
                                        con.register("uploaded_table", tmp_df)
                                        st.info(f"✅ Registered uploaded_table with {len(tmp_df)} rows and {len(tmp_df.columns)} columns")
                                    else:
                                        raise Exception("Failed to parse CSV with any strategy")
                                        
                                except Exception as e:
                                    st.error(f"❌ Failed to register CSV data: {e}")
                                    st.error("💡 **CSV Troubleshooting Tips:**")
                                    st.error("1. Check that all fields containing commas are properly quoted with double quotes")
                                    st.error("2. Ensure consistent use of commas as separators")
                                    st.error("3. Check line 13 specifically for formatting issues")
                                    st.error("4. Make sure there are no extra commas at the end of lines")
                                    df_for_run = None

                        # register document table if exists
                        text_for_doc = extracted_text or st.session_state.get("extracted_text", "")
                        if text_for_doc:
                            paras = [ln.strip() for ln in text_for_doc.splitlines() if ln.strip()]
                            if paras:
                                doc_df = pd.DataFrame({"paragraph_id": list(range(1, len(paras) + 1)), "text": paras})
                                try:
                                    con.register("document", doc_df)
                                except Exception:
                                    pass
                        
                        # If no uploaded_table was registered, create a simple test table
                        if df_for_run is None:
                            try:
                                # Create a simple test table for demonstration
                                test_data = {
                                    'id': [1, 2, 3, 4, 5],
                                    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
                                    'age': [25, 30, 35, 28, 32],
                                    'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney']
                                }
                                test_df = pd.DataFrame(test_data)
                                df_for_run = test_df
                                con.register("uploaded_table", test_df)
                                st.info("ℹ️ No file uploaded. Using sample data for demonstration.")
                            except Exception as e:
                                st.warning(f"⚠️ Failed to create test table: {e}")

                        # call wrapper
                        try:
                            df_res = generate_and_run_sql(query_prompt, df_for_run, con)
                            st.success("Local query executed — results below:")
                            st.dataframe(df_res)
                        except Exception as e:
                            st.error(f"Local SQL execution failed after automatic repair attempts: {e}")
            except Exception as e:
                st.error(f"SQL generation or execution failed: {e}")

st.markdown("---")
st.caption(
    "This app attempts iterative, automatic repairs for SQL errors: rule-based transforms, fuzzy mapping, numeric casts, and model rewrites targeted at DuckDB. It will try multiple candidates and return the first successful result. Review the Auto-Fix Log for details."
)

# show auto-fix log (if any)
_init_auto_fix_log()
if st.session_state.get("auto_fix_log"):
    with st.expander(f"Auto-Fix Log ({len(st.session_state['auto_fix_log'])} entries)", expanded=False):
        for i, e in enumerate(reversed(st.session_state["auto_fix_log"]), start=1):
            st.markdown(f"**Entry #{i} — {e.get('timestamp','-')}**")
            st.markdown(f"- **Type:** {e.get('type')}")
            if e.get("attempt"):
                st.markdown(f"- **Attempt:** {e.get('attempt')}")
            if e.get("reason"):
                st.markdown(f"- **Reason:** {e.get('reason')}")
            if e.get("error_class"):
                st.markdown(f"- **Error class:** {e.get('error_class')}")
            if e.get("error_message"):
                st.markdown(f"- **Error:** {e.get('error_message')}")
            if e.get("note"):
                st.markdown(f"- **Note:** {e.get('note')}")
            if e.get("original_sql"):
                st.markdown(f"- **Original SQL:**")
                st.code(e.get("original_sql","(none)"), language="sql")
            if e.get("candidate_sql"):
                st.markdown(f"- **Candidate SQL (attempt):**")
                st.code(e.get("candidate_sql","(none)"), language="sql")
            if e.get("fixed_sql"):
                st.markdown(f"- **Fixed SQL (successful):**")
                st.code(e.get("fixed_sql","(none)"), language="sql")
            st.markdown("---")
else:
    with st.expander("Auto-Fix Log (empty)", expanded=False):
        st.write("No automatic fixes applied yet.")
