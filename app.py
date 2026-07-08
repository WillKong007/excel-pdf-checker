import streamlit as st
import os
import tempfile
import zipfile
import io
import re
import logging
import unicodedata
import time
from collections import defaultdict
from pathlib import Path
import pandas as pd
import pdfplumber
import cv2
import pytesseract
from pdf2image import convert_from_path
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import range_boundaries
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
        'S': '5', 's': '5',
        'Z': '2', 'z': '2',
        'G': '6', 'g': '6',
        'T': '7', 't': '7',
        'B': '8', 'b': '8',
        'A': '4', 'a': '4',
        'E': '3', 'e': '3',
    }
    for old, new in mapping.items():
        text = text.replace(old, new)
    return text

# Selectively applies OCR correction ONLY to tokens that contain numbers
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
        #  Convert PDF page to image
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num
        )
        if not images:
            return page_num, ""
        # Convert to OpenCV format
        img = np.array(images[0])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Grayscale for faster processing
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #Apply adaptive thresholding
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
    "token_presence_threshold": 1.0,
    "force_ocr": True,
    "max_pages": 0,
    "ocr_dpi": 300,
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
        if max_pages == 0:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
        else:
            total_pages = max_pages
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

        if not all_text or len(''.join(all_text).strip()) < 100:
            logging.info("pdfplumber extracted little text; falling back to parallel OCR.")
            return self.extract_pdf_content_parallel(pdf_path, progress_callback)

        combined = ' '.join(all_text)
        combined = re.sub(r'\s+', ' ', combined).strip()
        return combined, all_text

    def tokenize_english(self, text):
        if not text:
            return []
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\u00a0', ' ')
        text = text.replace('\u00ad', '')
        text = re.sub(r'\s+', ' ', text)
        tokens = re.findall(r'[a-zA-Z0-9]+(?:[-\u2010-\u2015][a-zA-Z0-9]+)*', text)
        return [t.lower() for t in tokens if len(t) >= self.config["min_word_length"]]

    # --- UPDATED compare_content: REORDERED CHECKS ---
    def compare_content(self, excel_text, pdf_text, cell_map):
        excel_tokens = set(self.tokenize_english(excel_text))
        pdf_tokens = set(self.tokenize_english(pdf_text))
        missing_global = excel_tokens - pdf_tokens - self.ignore_tokens

        # --- UNIVERSAL FILTER (removes punctuation/spaces for comparison) ---
        pdf_clean = re.sub(r'[^a-zA-Z0-9]', '', pdf_text.lower())
        filtered_missing = set()
        def is_hyphenated_present(token, pdf_clean):
            parts = re.split(r'[-\u2010-\u2015\u2212\u00ad]', token)
            if len(parts) < 2:
                return False
            parts_clean = [re.sub(r'[^a-zA-Z0-9]', '', p) for p in parts if p]
            if not all(parts_clean):
                return False
            return all(p in pdf_clean for p in parts_clean)

        for token in missing_global:
            token_clean = re.sub(r'[^a-zA-Z0-9]', '', token)
            if token_clean in pdf_clean or safe_normalize(token_clean) in safe_normalize(pdf_clean):
                continue
            if any(c in token for c in '-\u2010\u2015\u2212\u00ad'):
                if is_hyphenated_present(token, pdf_clean):
                    continue
            filtered_missing.add(token)
        missing_global = filtered_missing
        # --------------------------------------------------------------

        missing_cells = defaultdict(list)
        threshold = self.config.get("token_presence_threshold", 1.0)

        # Prepare cleaned PDF for substring checks (once per call)
        pdf_no_space = re.sub(r'\s+', '', self.normalize_text(pdf_text))
        pdf_clean_sub = re.sub(r'[^a-zA-Z0-9]', '', pdf_no_space)

        for (sheet, ref), val in cell_map.items():
            tokens_in_cell = set(self.tokenize_english(val))
            if not tokens_in_cell:
                continue

            # ---- LAYER 1: CELL-LEVEL SEQUENCE CHECK (TRUNCATION DETECTION) ----
  
            cell_norm = self.normalize_text(val)
            cell_no_space = re.sub(r'\s+', '', cell_norm)
            cell_clean = re.sub(r'[^a-zA-Z0-9]', '', cell_no_space)

            # Check if the entire cleaned cell appears as a substring
            if self.config.get("use_substring_check", False):
                if cell_clean in pdf_clean_sub:
                    continue   # cell is fully present – skip it

            # ---- LAYER 2: PREFIX/SUFFIX CHECK (for long cells) ----
            # If the beginning exists but the end is missing, flag it immediately.
            if len(tokens_in_cell) > 10:
                prefix = cell_clean[:100]
                suffix = cell_clean[-100:]
                if prefix in pdf_clean_sub and suffix not in pdf_clean_sub:
                    # The cell is truncated at the end
                    missing_tail = cell_norm[-200:]
                    missing_cells[(sheet, ref)].append(f"TRUNCATED: {missing_tail}")
                    continue

            # ---- LAYER 3: RATIO CHECK (fallback for cells that pass the above) ----
            has_numbers = any(re.search(r'\d', t) for t in tokens_in_cell)

            present = tokens_in_cell & pdf_tokens
            ratio = len(present) / len(tokens_in_cell)

            if ratio < threshold:
                cell_missing = tokens_in_cell - pdf_tokens - self.ignore_tokens
                cell_missing_clean = {
                    t for t in cell_missing
                    if re.sub(r'[^a-zA-Z0-9]', '', t) not in pdf_clean
                    and safe_normalize(re.sub(r'[^a-zA-Z0-9]', '', t)) not in safe_normalize(pdf_clean)
                    and not is_hyphenated_present(t, pdf_clean)
                }
                if has_numbers:
                    cell_missing_clean = {t for t in cell_missing_clean if re.search(r'\d', t)}
                if cell_missing_clean:
                    missing_cells[(sheet, ref)].extend(list(cell_missing_clean))

        final_missing = set()
        for tokens in missing_cells.values():
            final_missing.update(tokens)

        # Global similarity (for reference)
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
            "missing_percentage": len(final_missing) / max(len(excel_tokens), 1) * 100,
            "similarity_score": similarity,
            "missing_tokens": list(final_missing),
            "missing_cells": dict(missing_cells),
            "pdf_text": pdf_text,
            "cell_map": cell_map,
        }
        return stats

    def check_pair(self, excel_path, pdf_path, progress_callback=None):
        logging.info(f"Checking: {excel_path} ↔ {pdf_path}")
        excel_text, _, cell_map = self.extract_excel_content(excel_path)
        pdf_text, _ = self.extract_pdf_content(pdf_path, progress_callback)
        if not excel_text or not pdf_text:
            logging.error("Could not extract content from one of the files.")
            return None
        stats = self.compare_content(excel_text, pdf_text, cell_map)
        stats["excel_file"] = os.path.basename(excel_path)
        stats["pdf_file"] = os.path.basename(pdf_path)
        self.results.append(stats)
        if stats["missing_percentage"] > self.config["missing_threshold"] * 100:
            logging.warning(f"ALERT: Missing {stats['missing_word_count']} words ({stats['missing_percentage']:.2f}%) in PDF.")
        logging.info(f"SUMMARY: Excel words: {stats['excel_word_count']}, PDF words: {stats['pdf_word_count']}, Missing: {stats['missing_word_count']} ({stats['missing_percentage']:.2f}%), Similarity: {stats['similarity_score']:.4f}")
        return stats

    def generate_full_report(self):
        lines = []
        lines.append("=" * 70)
        lines.append("ALPHANUMERIC (ENGLISH & NUMBERS) LOSS DETECTION REPORT (FULL)")
        lines.append("=" * 70)
        for idx, res in enumerate(self.results, 1):
            lines.append(f"\n📄 Pair {idx}: {res['excel_file']} ↔ {res['pdf_file']}")
            lines.append(f"   Excel words: {res['excel_word_count']}")
            lines.append(f"   PDF words  : {res['pdf_word_count']}")
            lines.append(f"   Missing    : {res['missing_word_count']} ({res['missing_percentage']:.2f}%)")
            lines.append(f"   Similarity : {res['similarity_score']:.4f}")
            if res['missing_tokens']:
                lines.append("   ALL missing tokens: " + ", ".join(res['missing_tokens']))
            if res['missing_cells']:
                lines.append("   Cells with missing words:")
                for (sheet, ref), tokens in res['missing_cells'].items():
                    lines.append(f"      Sheet '{sheet}', cell {ref}: {', '.join(tokens)}")
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

# ------------------------------------------------------------
# 2. CHINESE CHECKER
# ------------------------------------------------------------
ZH_CONFIG = {
    "missing_threshold": 0.02,
    "ignore_hidden": True,
    "use_print_area": True,
    "cell_presence_threshold": 0.95,
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
            print_area = None
            if self.config["use_print_area"] and ws.print_area:
                if isinstance(ws.print_area, str):
                    print_area = ws.print_area.split(',')[0]
                else:
                    print_area = ws.print_area[0]
            min_row, max_row = 1, ws.max_row
            min_col, max_col = 1, ws.max_column
            if print_area:
                try:
                    if '!' in print_area:
                        print_area = print_area.split('!')[-1]
                    min_col_b, min_row_b, max_col_b, max_row_b = range_boundaries(print_area)
                    min_row = min_row_b or 1
                    max_row = max_row_b or ws.max_row
                    min_col = min_col_b or 1
                    max_col = max_col_b or ws.max_column
                except Exception as e:
                    logging.warning(f"工作表 [{sheet_name}] 的列印區域 '{print_area}' 解析失敗，將掃描全表。錯誤: {e}")
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
            logging.warning(f"pdfplumber 提取失敗: {e}")
        try:
            images = convert_from_path(pdf_path, dpi=250)
            total = len(images)
            psm = self.config.get("ocr_psm_zh", 6)
            for idx, img in enumerate(images):
                gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                ocr_text = pytesseract.image_to_string(thresh, lang='chi_tra', config=f'--psm {psm}')
                if ocr_text:
                    all_text.append(self.ultimate_normalize(ocr_text))
                if progress_callback:
                    progress_callback(idx+1, total)
        except Exception as e:
            logging.warning(f"OCR 提取失敗（若未安裝 tesseract 繁簡中文包可忽略）: {e}")
        combined = ' '.join(all_text)
        return combined, all_text

    def tokenize_chinese(self, text):
        if not text:
            return []
        text = self.ultimate_normalize(text)
        return re.findall(r'[\u4e00-\u9fff]', text)

    def check_cell_presence(self, cell_text, pdf_text):
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
        if ratio < 0.85:
            missing = cell_tokens - pdf_tokens
            return False, f"遺漏字元：{'、'.join(list(missing)[:10])}"

        chunk_size = 4
        total_chunks = len(cell_chars) - chunk_size + 1
        if total_chunks > 0:
            missing_chunks = []
            for i in range(total_chunks):
                chunk = "".join(cell_chars[i:i+chunk_size])
                if chunk not in pdf_chars_stream:
                    missing_chunks.append(cell_chars[i+chunk_size-1])
            if missing_chunks and (len(missing_chunks) / total_chunks > 0.05):
                unique_missing = list(set(missing_chunks))
                return False, f"疑似截斷（缺少 {len(unique_missing)} 個獨特字元）"

        suffix_len = min(5, len(cell_chars))
        end_suffix = "".join(cell_chars[-suffix_len:])
        if end_suffix not in pdf_chars_stream:
            return False, f"結尾被裁切（缺少後綴：{end_suffix}）"

        lines = re.split(r'[\n\r]+', cell_text)
        bullet_lines = []
        for line in lines:
            line_clean = re.sub(r'^[-\•\●\*]\s*', '', line).strip()
            if line_clean:
                bullet_lines.append(line_clean)

        for line in bullet_lines:
            line_clean_chars = self.tokenize_chinese(line)
            if not line_clean_chars:
                continue
            line_clean_str = "".join(line_clean_chars)
            if line_clean_str not in pdf_chars_stream:
                return False, f"遺漏段落：{line[:30]}..."

        return True, ""

    # --- UPDATED compare_content for Chinese (REORDERED CHECKS) ---
    def compare_content(self, excel_text, pdf_text, cell_map):
        excel_tokens = set(self.tokenize_chinese(excel_text))
        pdf_tokens = set(self.tokenize_chinese(pdf_text))
        missing_global = excel_tokens - pdf_tokens

        missing_cells = defaultdict(list)
        truncated_cells = {}

        # Prepare PDF character stream for substring checks
        pdf_chars_stream = "".join(self.tokenize_chinese(pdf_text))

        threshold = self.config.get("cell_presence_threshold", 0.95)

        for (sheet, ref), val in cell_map.items():
            chars_in_cell = set(self.tokenize_chinese(val))
            if not chars_in_cell:
                continue

            # ---- LAYER 1: CELL-LEVEL SUBSTRING CHECK ----
            cell_clean = "".join(chars_in_cell)
            if cell_clean in pdf_chars_stream:
                continue

            # ---- LAYER 2: PREFIX/SUFFIX CHECK (for long cells) ----
            if len(chars_in_cell) > 10:
                prefix = cell_clean[:50]
                suffix = cell_clean[-50:]
                if prefix in pdf_chars_stream and suffix not in pdf_chars_stream:
                    truncated_cells[f"{sheet}!{ref}"] = f"結尾被裁切（缺少後綴：{suffix}）"
                    continue

            # ---- LAYER 3: RATIO CHECK (fallback) ----
            present = chars_in_cell & pdf_tokens
            ratio = len(present) / len(chars_in_cell)

            if ratio < threshold:
                cell_missing = chars_in_cell - pdf_tokens
                if cell_missing:
                    missing_cells[(sheet, ref)].extend(list(cell_missing))

            # Also run the existing truncation check (N-gram, line-by-line) as a fallback
            is_present, missing_part = self.check_cell_presence(val, pdf_text)
            if not is_present:
                truncated_cells[f"{sheet}!{ref}"] = missing_part

        final_missing = set()
        for tokens in missing_cells.values():
            final_missing.update(tokens)

        tfidf = TfidfVectorizer(analyzer='char')
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
            "missing_percentage": len(final_missing) / max(len(excel_tokens), 1) * 100,
            "similarity_score": similarity,
            "missing_tokens": list(final_missing),
            "missing_cells": dict(missing_cells),
            "truncated_cells": truncated_cells,
            "truncated_count": len(truncated_cells),
            "pdf_text": pdf_text,
            "cell_map": cell_map,
        }
        return stats

    def check_pair(self, excel_path, pdf_path, progress_callback=None):
        logging.info(f"🚀 開始執行獨立【中文】內容審查: {excel_path} ↔ {pdf_path}")
        excel_text, _, cell_map = self.extract_excel_content(excel_path)
        pdf_text, _ = self.extract_pdf_content(pdf_path, progress_callback)
        if not excel_text or not pdf_text:
            logging.error("無法從檔案中提取出有效的中文文字內容。")
            return None
        stats = self.compare_content(excel_text, pdf_text, cell_map)
        stats["excel_file"] = os.path.basename(excel_path)
        stats["pdf_file"] = os.path.basename(pdf_path)
        self.results.append(stats)
        if stats["missing_percentage"] > self.config["missing_threshold"] * 100:
            logging.warning(f"⚠️ 中文內容缺失警告: 在 PDF 中漏掉了 {stats['missing_word_count']} 個中文字元 ({stats['missing_percentage']:.2f}%). 請查看報告。")
        logging.info(f"檢查完成結果 SUMMARY: Excel中文字元數: {stats['excel_word_count']}, PDF中文字元數: {stats['pdf_word_count']}, 完全丟失中文字數: {stats['missing_word_count']} ({stats['missing_percentage']:.2f}%), 相似度: {stats['similarity_score']:.4f}, 被裁切的中文字儲存格數: {stats['truncated_count']}")
        return stats

    def generate_report(self):
        lines = []
        lines.append("=" * 70)
        lines.append("        Excel → PDF 獨立【中文】內容漏字與裁切審查報告")
        lines.append("=" * 70)
        for idx, res in enumerate(self.results, 1):
            lines.append(f"\n📄 檔案組 {idx}: {res['excel_file']} ↔ {res['pdf_file']}")
            lines.append(f"   Excel 漢字數 : {res['excel_word_count']}")
            lines.append(f"   PDF 漢字數   : {res['pdf_word_count']}")
            lines.append(f"   徹底遺漏字數 : {res['missing_word_count']} ({res['missing_percentage']:.2f}%)")
            lines.append(f"   漢字相似度   : {res['similarity_score']:.4f}")
            lines.append(f"   遭截斷中文字格: {res.get('truncated_count', 0)}")
            if res['missing_tokens']:
                lines.append("   PDF中完全消失的漢字: " + ", ".join(res['missing_tokens'][:20]))
            if res.get('truncated_cells'):
                lines.append("   🚨 檢測到【中文字串遭截斷/不連續】的儲存格列表：")
                for cell, missing in list(res['truncated_cells'].items()):
                    lines.append(f"      儲存格 {cell} -> 該格疑似缺失或不連續的漢字有: {missing}")
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------
def get_pdf_snippet(pdf_text, word, context=50):
    if not word:
        return ""
    lower_text = pdf_text.lower()
    word_lower = word.lower()
    idx = lower_text.find(word_lower)
    if idx == -1:
        clean_word = re.sub(r'[^a-zA-Z0-9]', '', word_lower)
        clean_pdf = re.sub(r'[^a-zA-Z0-9]', '', pdf_text.lower())
        idx = clean_pdf.find(clean_word)
        if idx == -1:
            return "Word not found in PDF context."
        return f"(found in cleaned text) ...{pdf_text[max(0, idx-30):min(len(pdf_text), idx+30)]}..."
    start = max(0, idx - context)
    end = min(len(pdf_text), idx + len(word) + context)
    return pdf_text[start:end]

def generate_excel_report(all_results):
    rows = []
    for lang, data in all_results.items():
        stats = data['stats']
        for (sheet, ref), tokens in stats['missing_cells'].items():
            rows.append({
                "Language": lang.capitalize(),
                "Sheet": sheet,
                "Cell": ref,
                "Missing Words": ", ".join(tokens),
                "Count": len(tokens),
                "Excel File": stats.get('excel_file', '')
            })
    df = pd.DataFrame(rows)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Missing Words')
    return output.getvalue()

# ------------------------------------------------------------
# 4. COMBINED RUNNER
# ------------------------------------------------------------
def run_check(excel_path, pdf_path, mode='both', progress_callback=None):
    results = {}
    if mode in ('english', 'both'):
        en_checker = ExcelPDFAlphanumericChecker()
        en_stats = en_checker.check_pair(excel_path, pdf_path, progress_callback)
        if en_stats:
            results['alphanumeric'] = {'stats': en_stats, 'report': en_checker.generate_full_report()}
    if mode in ('chinese', 'both'):
        zh_checker = ExcelPDFChineseChecker()
        zh_stats = zh_checker.check_pair(excel_path, pdf_path, progress_callback)
        if zh_stats:
            results['chinese'] = {'stats': zh_stats, 'report': zh_checker.generate_report()}
    return results

def process_zip(zip_file, mode='both'):
    results = []
    with zipfile.ZipFile(zip_file, 'r') as z:
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extractall(tmpdir)
            files = os.listdir(tmpdir)
            excel_files = [f for f in files if f.endswith('.xlsx')]
            pairs = []
            for ex in excel_files:
                base = os.path.splitext(ex)[0]
                pdf_candidates = [f for f in files if f.endswith('.pdf') and os.path.splitext(f)[0] == base]
                if pdf_candidates:
                    pairs.append((os.path.join(tmpdir, ex), os.path.join(tmpdir, pdf_candidates[0])))
            if not pairs:
                return [{"error": "No matching Excel-PDF pairs found in ZIP."}]
            for idx, (ex_path, pdf_path) in enumerate(pairs):
                def file_progress(completed, total):
                    pass
                result = run_check(ex_path, pdf_path, mode, file_progress)
                if result:
                    results.append({
                        "excel": os.path.basename(ex_path),
                        "pdf": os.path.basename(pdf_path),
                        "results": result
                    })
    return results

# ------------------------------------------------------------
# 5. STREAMLIT UI
# ------------------------------------------------------------
st.set_page_config(page_title="Excel ↔ PDF Missing Content Checker", layout="wide")
st.title("📄 Excel ↔ PDF Content Loss Detector")
st.markdown("Upload your files. The tool will compare them and report missing **Alphanumeric (English & Numbers)** or **Chinese** content.")

mode = st.radio(
    "Select checker mode:",
    ["Both (Alphanumeric + Chinese)", "Alphanumeric (English & Numbers) only", "Chinese only"],
    index=0
)
mode_map = {
    "Both (Alphanumeric + Chinese)": "both",
    "Alphanumeric (English & Numbers) only": "english",
    "Chinese only": "chinese"
}

with st.expander("⚙️ Advanced OCR Settings"):
    use_psm_11 = st.checkbox("Use PSM 11 (for PDFs without visible borders / tables)", value=False)
    if use_psm_11:
        EN_CONFIG["ocr_psm"] = 11
        ZH_CONFIG["ocr_psm_zh"] = 11
    else:
        EN_CONFIG["ocr_psm"] = 6
        ZH_CONFIG["ocr_psm_zh"] = 6

upload_type = st.radio("Upload mode:", ["Single File (Excel + PDF)", "Batch (ZIP with multiple pairs)"], index=0)

excel_file = None
pdf_file = None
zip_file = None

if upload_type == "Single File (Excel + PDF)":
    col1, col2 = st.columns(2)
    with col1:
        excel_file = st.file_uploader("Choose Excel file (.xlsx)", type=["xlsx"])
    with col2:
        pdf_file = st.file_uploader("Choose PDF file (.pdf)", type=["pdf"])
else:
    zip_file = st.file_uploader("Choose ZIP file containing Excel-PDF pairs", type=["zip"])
    st.caption("⚠️ ZIP must contain pairs with the same base name: e.g., `file1.xlsx` + `file1.pdf`, `file2.xlsx` + `file2.pdf`.")

if st.button("🔍 Check for Missing Content"):
    if upload_type == "Single File (Excel + PDF)":
        if not excel_file or not pdf_file:
            st.error("Please upload both Excel and PDF files.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_excel:
            tmp_excel.write(excel_file.read())
            excel_path = tmp_excel.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_file.read())
            pdf_path = tmp_pdf.name

        results = {}
        status = st.status("Running analysis...", expanded=True)
        progress_bar = st.progress(0)

        def progress_callback(completed, total):
            progress_bar.progress(completed / total)
            status.write(f"Processing page {completed} of {total}")

        with status:
            results = run_check(excel_path, pdf_path, mode_map[mode], progress_callback)

        status.update(label="✅ Analysis complete!", state="complete")

        try:
            os.unlink(excel_path)
        except FileNotFoundError:
            pass
        try:
            os.unlink(pdf_path)
        except FileNotFoundError:
            pass

    else:  # Batch mode
        if not zip_file:
            st.error("Please upload a ZIP file.")
            st.stop()

        status = st.status("Processing ZIP file...", expanded=True)
        progress_bar = st.progress(0)
        with status:
            batch_results = process_zip(zip_file, mode_map[mode])
            status.write(f"Processed {len(batch_results)} file pairs.")
        status.update(label="✅ Batch analysis complete!", state="complete")
        results = {}
        for item in batch_results:
            if "error" in item:
                st.error(item["error"])
            else:
                for lang, data in item["results"].items():
                    if lang not in results:
                        results[lang] = data

    # ---- DISPLAY RESULTS ----
    if not results:
        st.error("No content extracted from one of the files. Check file format.")
    else:
        st.success("✅ Analysis complete!")
        combined_report = ""
        display_names = {
            "alphanumeric": "Alphanumeric (English & Numbers)",
            "chinese": "Chinese"
        }

        all_missing_cells = {}
        for lang, data in results.items():
            display_name = display_names.get(lang, lang.capitalize())
            stats = data['stats']

            total_cells = len(stats['cell_map'])
            cells_with_issues = len(stats['missing_cells']) + len(stats.get('truncated_cells', {}))
            cell_issue_pct = (cells_with_issues / total_cells * 100) if total_cells > 0 else 0.0

            st.subheader(f"📊 {display_name} Checker Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Cells", total_cells)
            col2.metric("Cells with Issues", cells_with_issues)
            col3.metric("Issues % (Cells)", f"{cell_issue_pct:.2f}%")
            col4.metric("Similarity", f"{stats['similarity_score']:.4f}")

            # ---- Unified table for missing + truncated ----
            if stats['missing_cells'] or stats.get('truncated_cells'):
                rows = []
                for (sheet, ref), tokens in stats['missing_cells'].items():
                    rows.append({
                        "Sheet": sheet,
                        "Cell": ref,
                        "Missing Words": ", ".join(tokens),
                        "Count": len(tokens)
                    })
                for cell, msg in stats.get('truncated_cells', {}).items():
                    # Parse the message for Chinese truncations
                    missing_part = msg
                    if "缺少後綴：" in msg:
                        missing_part = msg.split("缺少後綴：")[1].rstrip("）")
                    elif "遺漏段落：" in msg:
                        missing_part = msg.split("遺漏段落：")[1].rstrip("...")
                    else:
                        if "：" in msg:
                            missing_part = msg.split("：")[1]
                    if "!" in cell:
                        sheet, ref = cell.split("!", 1)
                    else:
                        sheet, ref = "Unknown", cell
                    rows.append({
                        "Sheet": sheet,
                        "Cell": ref,
                        "Missing Words": missing_part,
                        "Count": len(missing_part)
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                all_missing_cells[lang] = stats['missing_cells']
            else:
                st.info(f"🎉 No missing or truncated {display_name} content detected!")

            # ---- Side-by-Side Preview ----
            if stats['missing_cells']:
                st.subheader(f"🔍 {display_name} Side-by-Side Cell Preview")
                for (sheet, ref), tokens in list(stats['missing_cells'].items())[:10]:
                    with st.expander(f"📌 Sheet '{sheet}', Cell {ref} (missing {len(tokens)} words)"):
                        col1, col2 = st.columns(2)
                        excel_text = stats['cell_map'].get((sheet, ref), "N/A")
                        with col1:
                            st.markdown("**📗 Excel Text:**")
                            st.text_area("", excel_text, height=150, key=f"excel_{sheet}_{ref}")
                        with col2:
                            st.markdown("**📘 PDF Snippet (first missing word context):**")
                            sample_token = tokens[0] if tokens else ""
                            snippet = get_pdf_snippet(stats['pdf_text'], sample_token)
                            st.text_area("", snippet, height=150, key=f"pdf_{sheet}_{ref}")

            report = data['report']
            combined_report += f"\n{'='*70}\n{display_name.upper()} CHECKER REPORT\n{'='*70}\n{report}\n"
            st.text_area(f"{display_name} Full Report", report, height=200)

        # ---- Download buttons ----
        if all_missing_cells:
            excel_data = generate_excel_report(results)
            st.download_button(
                label="⬇️ Download Excel Report (.xlsx)",
                data=excel_data,
                file_name="missing_words_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.download_button(
            label="⬇️ Download Full Combined Report (.txt)",
            data=combined_report,
            file_name="combined_missing_words_report.txt",
            mime="text/plain"
        )

st.markdown("---")
st.caption("Built with Streamlit | Processing runs on Google Colab backend.")
