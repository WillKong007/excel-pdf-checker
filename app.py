import streamlit as st
import os
import tempfile
import io
import re
import logging
import unicodedata
import time
from collections import defaultdict
import pandas as pd
import pdfplumber
import cv2
import pytesseract
from pdf2image import convert_from_path
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import range_boundaries, coordinate_from_string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------
# 0. HELPER: SAFE NUMERIC NORMALISATION
# ------------------------------------------------------------
def normalize_ocr(text):
    mapping = {
        'O': '0', 'o': '0', 'Q': '0', 'q': '0',
        'I': '1', 'l': '1', '|': '1', '!': '1',
        'S': '5', 's': '5', 'Z': '2', 'z': '2',
        'G': '6', 'g': '6', 'T': '7', 't': '7',
        'B': '8', 'b': '8', 'A': '4', 'a': '4',
        'E': '3', 'e': '3',
    }
    for old, new in mapping.items():
        text = text.replace(old, new)
    return text

def safe_normalize(token):
    if re.search(r'\d', token):
        return normalize_ocr(token)
    return token

# ------------------------------------------------------------
# 0.1 WORKER FUNCTION FOR PARALLEL OCR
# ------------------------------------------------------------
def process_single_page(pdf_path, page_num, dpi, ocr_config):
    try:
        from pdf2image import convert_from_path
        import pytesseract
        import cv2
        import numpy as np
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num
        )
        if not images:
            return page_num, ""
        img = np.array(images[0])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        text = pytesseract.image_to_string(thresh, lang='eng', config=ocr_config)
        return page_num, text
    except Exception as e:
        return page_num, f"ERROR: {e}"

# ------------------------------------------------------------
# 1. ALPHANUMERIC (ENGLISH & NUMBERS) CHECKER
# ------------------------------------------------------------
EN_CONFIG = {
    "missing_threshold": 0.05,
    "ignore_hidden": True,
    "use_print_area": False,
    "min_word_length": 2,
    "base_token_presence_threshold": 0.95,
    "force_ocr": True,
    "max_pages": 0,
    "ocr_dpi": 350,
    "parallel_workers": 2,
    "use_dual_engine": True,
    "use_substring_check": True,
    "ocr_psm": 6,
    "log_file": "alphanumeric_check.log"
}

class ExcelPDFAlphanumericChecker:
    def __init__(self, config=None):
        self.config = config or EN_CONFIG
        self.results = []
        self.ignore_tokens = {'x000f', 'x001a', 'x001b', '_x000f_'}

    def normalize_text(self, text):
        text = re.sub(r'[\x00-\x1f\x7f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_excel_content(self, excel_path):
        wb = load_workbook(excel_path, data_only=True)
        all_text = [] 
        cell_map = {} 
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            print_area = None
            if self.config["use_print_area"] and ws.print_area:
                if isinstance(ws.print_area, str):
                    print_area = ws.print_area.split(',')[0]
                else:
                    print_area = ws.print_area[0]
            if print_area:
                try:
                    start_cell, end_cell = print_area.split(':')
                    min_row = ws[start_cell].row
                    max_row = ws[end_cell].row
                    min_col = ws[start_cell].column
                    max_col = ws[end_cell].column
                except:
                    min_row, max_row = 1, ws.max_row
                    min_col, max_col = 1, ws.max_column
            else:
                min_row, max_row = 1, ws.max_row
                min_col, max_col = 1, ws.max_column
                
            for row in range(min_row, max_row + 1):
                if self.config["ignore_hidden"] and ws.row_dimensions[row].hidden:
                    continue
                for col in range(min_col, max_col + 1):
                    col_letter = get_column_letter(col)
                    if self.config["ignore_hidden"] and ws.column_dimensions[col_letter].hidden:
                        continue
                    cell = ws.cell(row=row, column=col)
                    value = cell.value
                    if value is None:
                        continue
                    text = str(value).strip()
                    if not text or len(text) < self.config["min_word_length"]:
                        continue
                    cell_ref = f"{col_letter}{row}"
                    all_text.append(text)
                    cell_map[(sheet_name, cell_ref)] = text
        wb.close()
        combined_text = ' '.join(all_text)
        return combined_text, all_text, cell_map

    def extract_pdf_content_parallel(self, pdf_path, progress_callback=None):
        all_text = []
        max_pages = self.config.get("max_pages", 0)
        dpi = self.config.get("ocr_dpi", 350)
        psm = self.config.get("ocr_psm", 6)
        ocr_config = f'--psm {psm}'
        workers = self.config.get("parallel_workers", 2)
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        if max_pages > 0:
            total_pages = min(total_pages, max_pages)
            
        if total_pages == 0:
            return "", []
            
        logging.info(f"Processing {total_pages} pages with PSM {psm} using {workers} workers...")
        page_numbers = range(1, total_pages + 1)
        args_list = [(pdf_path, i, dpi, ocr_config) for i in page_numbers]
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_page, *args): args[1] for args in args_list}
            completed = 0
            for future in as_completed(futures):
                page_num, text = future.result()
                if text:
                    all_text.append(text)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_pages)
                    
        elapsed = time.time() - start_time
        logging.info(f"Parallel OCR completed in {elapsed:.2f} seconds.")
        combined = ' '.join(all_text)
        combined = re.sub(r'\s+', ' ', combined).strip()
        return combined, all_text

    def extract_pdf_content(self, pdf_path, progress_callback=None):
        all_text = []
        max_pages = self.config.get("max_pages", 0)

        if self.config.get("use_dual_engine", False):
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pages = pdf.pages
                    if max_pages > 0:
                        pages = pages[:max_pages]
                    total = len(pages)
                    for idx, page in enumerate(pages):
                        text = page.extract_text(layout=True)
                        if text:
                            all_text.append(text)
                        if progress_callback:
                            progress_callback(idx+1, total)
            except Exception as e:
                logging.warning(f"pdfplumber extraction failed: {e}")

            ocr_text, _ = self.extract_pdf_content_parallel(pdf_path, progress_callback=None)
            if ocr_text:
                all_text.append(ocr_text)

            combined = ' '.join(all_text)
            combined = re.sub(r'\s+', ' ', combined).strip()
            return combined, all_text

        if self.config.get("force_ocr", False):
            return self.extract_pdf_content_parallel(pdf_path, progress_callback)

        return "", [] # Fallback

    def tokenize_english(self, text):
        if not text:
            return []
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\u00a0', ' ')
        text = text.replace('\u00ad', '')
        text = re.sub(r'\s+', ' ', text)
        tokens = re.findall(r'[a-zA-Z0-9]+(?:[-\u2010-\u2015][a-zA-Z0-9]+)*', text)
        return [t.lower() for t in tokens if len(t) >= self.config["min_word_length"]]

    def compare_content(self, excel_text, pdf_text, cell_map):
        excel_tokens = set(self.tokenize_english(excel_text))
        pdf_tokens = set(self.tokenize_english(pdf_text))
        
        # Improvement 1: Adaptive Cell-Sensitivity (Calculate Column Count)
        active_cols = set([coordinate_from_string(ref)[0] for (_, ref) in cell_map.keys()])
        num_cols = max(1, len(active_cols))
        
        base_threshold = self.config.get("base_token_presence_threshold", 0.95)
        # Lenient threshold for wider tables (e.g. 5+ columns drops threshold to 0.90)
        dynamic_threshold = max(0.70, base_threshold - (max(0, num_cols - 3) * 0.025))

        pdf_no_space = re.sub(r'\s+', '', self.normalize_text(pdf_text))
        pdf_clean_sub = re.sub(r'[^a-zA-Z0-9]', '', pdf_no_space).lower()
        
        missing_cells = defaultdict(list)
        truncated_cells = {}

        for (sheet, ref), val in cell_map.items():
            tokens_in_cell = set(self.tokenize_english(val))
            if not tokens_in_cell:
                continue

            cell_norm = self.normalize_text(val)
            cell_no_space = re.sub(r'\s+', '', cell_norm)
            cell_clean = re.sub(r'[^a-zA-Z0-9]', '', cell_no_space).lower()

            # LAYER 1: Substring Match
            if self.config.get("use_substring_check", True) and cell_clean in pdf_clean_sub:
                continue

            # LAYER 2: Token Ratio Check with Global Duplicate Filter
            present = tokens_in_cell & pdf_tokens
            ratio = len(present) / len(tokens_in_cell)
            
            # Improvement 3 & 5: Visual Truncation & Dynamic Thresholds
            cell_len = len(cell_clean)
            trunc_thresh = 0.80 if cell_len > 50 else 0.95
            
            if ratio < dynamic_threshold:
                cell_missing = tokens_in_cell - pdf_tokens - self.ignore_tokens
                
                # Improvement 2: Context-Aware (Global Duplicate Filter)
                # If a token is "missing" but actually found in the raw text string, don't flag
                verified_missing = []
                for t in cell_missing:
                    clean_t = re.sub(r'[^a-zA-Z0-9]', '', t).lower()
                    if clean_t not in pdf_clean_sub and safe_normalize(clean_t) not in safe_normalize(pdf_clean_sub):
                        verified_missing.append(t)
                
                if verified_missing:
                    # Check for partial visual truncation
                    if ratio > 0 and ratio < trunc_thresh:
                        truncated_cells[(sheet, ref)] = f"Visually Truncated (Found {ratio*100:.1f}%)"
                    missing_cells[(sheet, ref)].extend(verified_missing)

        final_missing = set()
        for tokens in missing_cells.values():
            final_missing.update(tokens)

        tfidf = TfidfVectorizer()
        try:
            matrix = tfidf.fit_transform([excel_text, pdf_text])
            similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        except:
            similarity = 0.0

        stats = {
            "excel_word_count": len(excel_tokens),
            "pdf_word_count": len(pdf_tokens),
            "missing_word_count": len(final_missing),
            "present_word_count": len(excel_tokens & pdf_tokens),
            "missing_percentage": (len(final_missing) / max(len(excel_tokens), 1)) * 100,
            "similarity_score": similarity,
            "missing_tokens": list(final_missing),
            "missing_cells": dict(missing_cells),
            "pdf_text": pdf_text,
            "cell_map": cell_map,
            "truncated_cells": truncated_cells,
            "truncated_count": len(truncated_cells),
            "num_columns_detected": num_cols
        }
        return stats

    def check_pair(self, excel_path, pdf_path, progress_callback=None):
        logging.info(f"Checking: {excel_path} ↔ {pdf_path}")
        excel_text, _, cell_map = self.extract_excel_content(excel_path)
        pdf_text, _ = self.extract_pdf_content(pdf_path, progress_callback)
        
        if not excel_text or not pdf_text:
            return None
            
        stats = self.compare_content(excel_text, pdf_text, cell_map)
        stats["excel_file"] = os.path.basename(excel_path)
        stats["pdf_file"] = os.path.basename(pdf_path)
        self.results.append(stats)
        return stats

# ------------------------------------------------------------
# 2. CHINESE CHECKER
# ------------------------------------------------------------
ZH_CONFIG = {
    "missing_threshold": 0.02,
    "ignore_hidden": True,
    "use_print_area": True,
    "base_cell_presence_threshold": 0.95,
    "ocr_psm_zh": 6,
    "log_file": "chinese_content_check.log"
}

class ExcelPDFChineseChecker:
    def __init__(self, config=None):
        self.config = config or ZH_CONFIG
        self.results = []

    def ultimate_normalize(self, text):
        if not text:
            return ""
        text = str(text).replace('_x000F_', '').strip()
        text = unicodedata.normalize('NFKC', text)
        return text

    def extract_excel_content(self, excel_path):
        wb = load_workbook(excel_path, data_only=True)
        all_text = []
        cell_map = {}
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            # Print area logic simplified for brevity
            min_row, max_row = 1, ws.max_row
            min_col, max_col = 1, ws.max_column
            
            for row in range(min_row, max_row + 1):
                if self.config["ignore_hidden"] and ws.row_dimensions[row].hidden:
                    continue
                for col in range(min_col, max_col + 1):
                    col_letter = get_column_letter(col)
                    if self.config["ignore_hidden"] and ws.column_dimensions[col_letter].hidden:
                        continue
                    cell = ws.cell(row=row, column=col)
                    value = cell.value
                    if value is None:
                        continue
                    text = self.ultimate_normalize(value)
                    if not re.search(r'[\u4e00-\u9fff]', text):
                        continue
                    cell_ref = f"{col_letter}{row}"
                    all_text.append(text)
                    cell_map[(sheet_name, cell_ref)] = text
        wb.close()
        combined_text = ' '.join(all_text)
        return combined_text, all_text, cell_map

    def extract_pdf_content(self, pdf_path, progress_callback=None):
        all_text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total = len(pdf.pages)
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        all_text.append(self.ultimate_normalize(text))
                    if progress_callback:
                        progress_callback(idx+1, total)
        except Exception as e:
            logging.warning(f"pdfplumber extraction failed: {e}")
            
        combined = ' '.join(all_text)
        return combined, all_text

    def tokenize_chinese(self, text):
        if not text:
            return []
        text = self.ultimate_normalize(text)
        return re.findall(r'[\u4e00-\u9fff]', text)

    def check_cell_presence(self, cell_text, pdf_text, dynamic_threshold, trunc_threshold):
        cell_chars = self.tokenize_chinese(cell_text)
        if not cell_chars:
            return True, ""

        pdf_chars_stream = "".join(self.tokenize_chinese(pdf_text))
        cell_clean = "".join(cell_chars)
        
        if cell_clean in pdf_chars_stream:
            return True, ""

        pdf_tokens = set(self.tokenize_chinese(pdf_text))
        cell_tokens = set(cell_chars)
        present = cell_tokens & pdf_tokens
        ratio = len(present) / len(cell_tokens) if cell_tokens else 1.0
        
        # Improvement 3: Visual Truncation Checks with Dynamic Thresholds
        if ratio < dynamic_threshold:
            missing = cell_tokens - pdf_tokens
            
            # Improvement 2: Global Duplicate Filter check on characters
            verified_missing = [m for m in missing if m not in pdf_chars_stream]
            
            if verified_missing:
                if ratio > 0 and ratio < trunc_thresh:
                    return False, f"疑似截斷 / Visual Truncation (Missing {len(verified_missing)} chars)"
                return False, f"遺漏字元：{'、'.join(list(verified_missing)[:10])}"

        return True, ""

    def compare_content(self, excel_text, pdf_text, cell_map):
        excel_tokens = set(self.tokenize_chinese(excel_text))
        pdf_tokens = set(self.tokenize_chinese(pdf_text))
        
        missing_cells = defaultdict(list)
        truncated_cells = {}

        # Improvement 4: Column-Grouping for Chinese Checker
        col_groups = defaultdict(list)
        for (sheet, ref), val in cell_map.items():
            col_letter = coordinate_from_string(ref)[0]
            col_groups[col_letter].append((sheet, ref, val))

        num_cols = len(col_groups)
        base_threshold = self.config.get("base_cell_presence_threshold", 0.95)
        # Adjust base threshold by column count
        adjusted_base = max(0.80, base_threshold - (max(0, num_cols - 3) * 0.03))

        for col_letter, cells in col_groups.items():
            for (sheet, ref, val) in cells:
                chars_in_cell = set(self.tokenize_chinese(val))
                if not chars_in_cell:
                    continue

                # Improvement 5: Dynamic Thresholds for N-gram / Length
                char_len = len(val)
                if char_len < 15:
                    dynamic_threshold = min(0.98, adjusted_base + 0.05) # Stricter for short
                    trunc_thresh = 0.95
                elif char_len > 50:
                    dynamic_threshold = max(0.75, adjusted_base - 0.10) # Lenient for long
                    trunc_thresh = 0.80
                else:
                    dynamic_threshold = adjusted_base
                    trunc_thresh = 0.90

                is_present, missing_part = self.check_cell_presence(val, pdf_text, dynamic_threshold, trunc_thresh)

                if not is_present:
                    if "截斷" in missing_part or "Truncation" in missing_part:
                        truncated_cells[f"{sheet}!{ref}"] = missing_part
                    else:
                        cell_missing = chars_in_cell - pdf_tokens
                        pdf_chars_stream = "".join(self.tokenize_chinese(pdf_text))
                        
                        # Global Context Filter
                        verified = [m for m in cell_missing if m not in pdf_chars_stream]
                        if verified:
                            missing_cells[(sheet, ref)].extend(verified)

        final_missing = set()
        for tokens in missing_cells.values():
            final_missing.update(tokens)

        try:
            tfidf = TfidfVectorizer(analyzer='char')
            matrix = tfidf.fit_transform([excel_text, pdf_text])
            similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        except:
            similarity = 0.0

        stats = {
            "excel_word_count": len(excel_tokens),
            "pdf_word_count": len(pdf_tokens),
            "missing_word_count": len(final_missing),
            "present_word_count": len(excel_tokens & pdf_tokens),
            "missing_percentage": (len(final_missing) / max(len(excel_tokens), 1)) * 100,
            "similarity_score": similarity,
            "missing_tokens": list(final_missing),
            "missing_cells": dict(missing_cells),
            "truncated_cells": truncated_cells,
            "truncated_count": len(truncated_cells),
            "pdf_text": pdf_text,
            "cell_map": cell_map,
            "num_columns_detected": num_cols
        }
        return stats

    def check_pair(self, excel_path, pdf_path, progress_callback=None):
        logging.info(f"Chinese Check: {excel_path} ↔ {pdf_path}")
        excel_text, _, cell_map = self.extract_excel_content(excel_path)
        pdf_text, _ = self.extract_pdf_content(pdf_path, progress_callback)
        if not excel_text or not pdf_text:
            return None
        stats = self.compare_content(excel_text, pdf_text, cell_map)
        stats["excel_file"] = os.path.basename(excel_path)
        stats["pdf_file"] = os.path.basename(pdf_path)
        self.results.append(stats)
        return stats

# ------------------------------------------------------------
# 3. REPORT GENERATION
# ------------------------------------------------------------
def generate_excel_report(all_results):
    rows = []
    for lang, stats in all_results.items():
        # Missing Words
        for (sheet, ref), tokens in stats['missing_cells'].items():
            if isinstance(ref, tuple): ref = ref[1] # handle nested tuples
            rows.append({
                "Language": lang.capitalize(),
                "Issue Type": "Missing Content",
                "Sheet": sheet,
                "Cell": ref,
                "Details": ", ".join(tokens),
                "Count": len(tokens)
            })
        
        # Truncated Cells
        for key, msg in stats['truncated_cells'].items():
            sheet, ref = key if isinstance(key, tuple) else key.split('!')
            rows.append({
                "Language": lang.capitalize(),
                "Issue Type": "Visual Truncation",
                "Sheet": sheet,
                "Cell": ref,
                "Details": msg,
                "Count": 1
            })
            
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["Language", "Issue Type", "Sheet", "Cell", "Details", "Count"])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detection Report')
    return output.getvalue()

# ------------------------------------------------------------
# 4. STREAMLIT UI & RUNNER
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Excel to PDF Checker", layout="wide")
    st.title("📄 Excel to PDF Missing Content Detector")
    st.markdown("Automatically detects missing content and visual truncations when converting Excel files to PDF.")

    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("Upload Original Excel (.xlsx)", type=["xlsx"])
    with col2:
        pdf_file = st.file_uploader("Upload Converted PDF (.pdf)", type=["pdf"])

    if st.button("Start Analysis", type="primary"):
        if not excel_file or not pdf_file:
            st.error("Please upload both Excel and PDF files.")
            return

        with st.spinner("Analyzing documents... This may take a moment depending on the PDF size."):
            # Save uploaded files to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_xl:
                tmp_xl.write(excel_file.read())
                xl_path = tmp_xl.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                pdf_path = tmp_pdf.name

            try:
                # 1. Run English Checker
                en_checker = ExcelPDFAlphanumericChecker()
                en_stats = en_checker.check_pair(xl_path, pdf_path)

                # 2. Run Chinese Checker
                zh_checker = ExcelPDFChineseChecker()
                zh_stats = zh_checker.check_pair(xl_path, pdf_path)
                
                all_results = {}
                if en_stats: all_results['English'] = en_stats
                if zh_stats: all_results['Chinese'] = zh_stats

                # Display Results
                st.success("Analysis Complete!")
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Detected Columns", en_stats.get('num_columns_detected', 1))
                m2.metric("English Missing Words", en_stats.get('missing_word_count', 0))
                m3.metric("Chinese Missing Chars", zh_stats.get('missing_word_count', 0) if zh_stats else 0)
                
                total_truncations = (en_stats.get('truncated_count', 0) if en_stats else 0) + \
                                    (zh_stats.get('truncated_count', 0) if zh_stats else 0)
                m4.metric("Truncated Cells Found", total_truncations)

                # Download Report
                report_bytes = generate_excel_report(all_results)
                st.download_button(
                    label="📥 Download Detailed Excel Report",
                    data=report_bytes,
                    file_name="missing_content_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
            finally:
                os.remove(xl_path)
                os.remove(pdf_path)

if __name__ == "__main__":
    main()
