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
# 1. ALPHANUMERIC (ENGLISH & NUMBERS) CHECKER CONFIG & CLASS
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
    "parallel_workers": 2,  # optimized for Colab Free
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

    def _process_single_page(self, pdf_path, page_num, dpi, ocr_config, progress_callback=None):
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
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(thresh, lang='eng', config=ocr_config)
            return page_num, text
        except Exception as e:
            return page_num, f"ERROR: {e}"

    def extract_pdf_content_parallel(self, pdf_path, progress_callback=None):
        all_text = []
        max_pages = self.config.get("max_pages", 0)
        dpi = self.config.get("ocr_dpi", 300)
        ocr_config = '--psm 6'
        workers = self.config.get("parallel_workers", 2)
        if max_pages == 0:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
        else:
            total_pages = max_pages
        if total_pages == 0:
            return "", []
        logging.info(f"Processing {total_pages} pages in parallel with {workers} workers...")
        page_numbers = range(1, total_pages + 1)
        args_list = [(pdf_path, i, dpi, ocr_config) for i in page_numbers]
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._process_single_page, *args): args[1] for args in args_list}
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

    def compare_content(self, excel_text, pdf_text, cell_map):
        excel_tokens = set(self.tokenize_english(excel_text))
        pdf_tokens = set(self.tokenize_english(pdf_text))
        missing_global = excel_tokens - pdf_tokens - self.ignore_tokens
        pdf_clean = re.sub(r'[^a-zA-Z0-9]', '', pdf_text.lower())
        filtered_missing = set()
        for token in missing_global:
            token_clean = re.sub(r'[^a-zA-Z0-9]', '', token)
            if token_clean not in pdf_clean:
                filtered_missing.add(token)
        missing_global = filtered_missing
        missing_cells = defaultdict(list)
        threshold = self.config.get("token_presence_threshold", 1.0)
        for (sheet, ref), val in cell_map.items():
            tokens_in_cell = set(self.tokenize_english(val))
            if not tokens_in_cell:
                continue
            present = tokens_in_cell & pdf_tokens
            ratio = len(present) / len(tokens_in_cell)
            if ratio < threshold:
                cell_missing = tokens_in_cell - pdf_tokens - self.ignore_tokens
                cell_missing_clean = {
                    t for t in cell_missing
                    if re.sub(r'[^a-zA-Z0-9]', '', t) not in pdf_clean
                }
                if cell_missing_clean:
                    missing_cells[(sheet, ref)].extend(list(cell_missing_clean))
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
# 2. CHINESE CHECKER CONFIG & CLASS (SAME AS BEFORE)
# ------------------------------------------------------------
ZH_CONFIG = {
    "missing_threshold": 0.02,
    "ignore_hidden": True,
    "use_print_area": True,
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
            images = convert_from_path(pdf_path, dpi=200)
            total = len(images)
            for idx, img in enumerate(images):
                gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                ocr_text = pytesseract.image_to_string(thresh, lang='chi_sim')
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
        if not cell_chars or len(cell_chars) < 5:
            return True, ""
        pdf_chars_stream = "".join(self.tokenize_chinese(pdf_text))
        chunk_size = 4
        missing_chunks = []
        for i in range(len(cell_chars) - chunk_size + 1):
            chunk = "".join(cell_chars[i:i+chunk_size])
            if chunk not in pdf_chars_stream:
                missing_chunks.append(cell_chars[i+chunk_size-1])
        if missing_chunks:
            unique_missing = []
            for char in missing_chunks:
                if char not in unique_missing:
                    unique_missing.append(char)
            return False, "、".join(unique_missing[:15])
        return True, ""

    def compare_content(self, excel_text, pdf_text, cell_map):
        excel_tokens = set(self.tokenize_chinese(excel_text))
        pdf_tokens = set(self.tokenize_chinese(pdf_text))
        missing = excel_tokens - pdf_tokens
        present = excel_tokens & pdf_tokens
        missing_cells = defaultdict(list)
        for (sheet, ref), val in cell_map.items():
            chars_in_cell = self.tokenize_chinese(val)
            for char in chars_in_cell:
                if char in missing:
                    missing_cells[(sheet, ref)].append(char)
        truncated_cells = {}
        for (sheet, ref), cell_content in cell_map.items():
            is_present, missing_part = self.check_cell_presence(cell_content, pdf_text)
            if not is_present:
                truncated_cells[f"{sheet}!{ref}"] = missing_part
        tfidf = TfidfVectorizer(analyzer='char')
        try:
            matrix = tfidf.fit_transform([excel_text, pdf_text])
            similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        except:
            similarity = 0.0
        stats = {
            "excel_word_count": len(excel_tokens),
            "pdf_word_count": len(pdf_tokens),
            "missing_word_count": len(missing),
            "present_word_count": len(present),
            "missing_percentage": len(missing) / max(len(excel_tokens), 1) * 100,
            "similarity_score": similarity,
            "missing_tokens": list(missing)[:50],
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
# 3. HELPER FUNCTIONS (Side-by-Side, Excel Export, ZIP processing)
# ------------------------------------------------------------
def get_pdf_snippet(pdf_text, word, context=50):
    """Find a snippet of PDF text around a word (case-insensitive)."""
    if not word:
        return ""
    lower_text = pdf_text.lower()
    word_lower = word.lower()
    idx = lower_text.find(word_lower)
    if idx == -1:
        # Try cleaned version
        clean_word = re.sub(r'[^a-zA-Z0-9]', '', word_lower)
        clean_pdf = re.sub(r'[^a-zA-Z0-9]', '', pdf_text.lower())
        idx = clean_pdf.find(clean_word)
        if idx == -1:
            return "Word not found in PDF context."
        # Map back to original text roughly (approximate)
        return f"(found in cleaned text) ...{pdf_text[max(0, idx-30):min(len(pdf_text), idx+30)]}..."
    start = max(0, idx - context)
    end = min(len(pdf_text), idx + len(word) + context)
    return pdf_text[start:end]

def generate_excel_report(all_results):
    """Generate an Excel file from results."""
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
# 4. COMBINED RUNNER (with progress callback)
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
    """Extract ZIP and process each Excel-PDF pair."""
    results = []
    with zipfile.ZipFile(zip_file, 'r') as z:
        # Extract to temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extractall(tmpdir)
            # Find pairs: .xlsx and .pdf with same base name
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
                # Progress per file
                def file_progress(completed, total):
                    # You could update a global progress here, but we'll just log.
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

# ---- Mode Selection ----
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

# ---- File Input (Single vs Batch) ----
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

# ---- Run Button ----
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
        # Progress bar
        status = st.status("Running analysis...", expanded=True)
        progress_bar = st.progress(0)
        
        def progress_callback(completed, total):
            progress_bar.progress(completed / total)
            status.write(f"Processing page {completed} of {total}")
        
        with status:
            results = run_check(excel_path, pdf_path, mode_map[mode], progress_callback)
        
        status.update(label="✅ Analysis complete!", state="complete")
        os.unlink(excel_path)
        os.unlink(pdf_path)
        
    else:  # Batch mode
        if not zip_file:
            st.error("Please upload a ZIP file.")
            st.stop()
        
        status = st.status("Processing ZIP file...", expanded=True)
        progress_bar = st.progress(0)
        # We'll run batch processing; we don't have per-file fine progress here, but we can show count.
        with status:
            batch_results = process_zip(zip_file, mode_map[mode])
            status.write(f"Processed {len(batch_results)} file pairs.")
        status.update(label="✅ Batch analysis complete!", state="complete")
        # Combine results for display
        results = {}
        for item in batch_results:
            if "error" in item:
                st.error(item["error"])
            else:
                for lang, data in item["results"].items():
                    if lang not in results:
                        results[lang] = data
        # Note: in batch, we only show combined stats; fine for demo.

    # ---- Display Results ----
    if not results:
        st.error("No content extracted from one of the files. Check file format.")
    else:
        st.success("✅ Analysis complete!")
        combined_report = ""
        display_names = {
            "alphanumeric": "Alphanumeric (English & Numbers)",
            "chinese": "Chinese"
        }
        
        # Store all missing cells for Excel export and side-by-side
        all_missing_cells = {}
        for lang, data in results.items():
            display_name = display_names.get(lang, lang.capitalize())
            stats = data['stats']
            
            # ---- 2.A: Heatmap / Interactive Table ----
            st.subheader(f"📊 {display_name} Checker Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"{display_name} Excel words", stats['excel_word_count'])
            col2.metric(f"{display_name} PDF words", stats['pdf_word_count'])
            col3.metric(f"Missing {display_name}", stats['missing_word_count'])
            col4.metric("Missing %", f"{stats['missing_percentage']:.2f}%")
            
            # Table of missing cells (2.A Heatmap)
            if stats['missing_cells']:
                rows = []
                for (sheet, ref), tokens in stats['missing_cells'].items():
                    rows.append({"Sheet": sheet, "Cell": ref, "Missing Words": ", ".join(tokens), "Count": len(tokens)})
                df = pd.DataFrame(rows)
                # Color code the Count column (heatmap)
                st.dataframe(
                    df,
                    column_config={
                        "Count": st.column_config.NumberColumn(
                            "Missing Count",
                            help="Number of missing tokens in this cell",
                            format="%d",
                            min_value=0,
                            max_value=max(df["Count"]) if not df.empty else 1,
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
                all_missing_cells[lang] = stats['missing_cells']
            else:
                st.info(f"🎉 No missing {display_name} content detected!")
            
            # ---- 2.B: Side-by-Side Cell Preview ----
            if stats['missing_cells']:
                st.subheader(f"🔍 {display_name} Side-by-Side Cell Preview")
                for (sheet, ref), tokens in list(stats['missing_cells'].items())[:10]:  # limit to 10 for readability
                    with st.expander(f"📌 Sheet '{sheet}', Cell {ref} (missing {len(tokens)} words)"):
                        col1, col2 = st.columns(2)
                        # Left: Excel text
                        excel_text = stats['cell_map'].get((sheet, ref), "N/A")
                        with col1:
                            st.markdown("**📗 Excel Text:**")
                            st.text_area("", excel_text, height=150, key=f"excel_{sheet}_{ref}")
                        # Right: PDF snippet
                        with col2:
                            st.markdown("**📘 PDF Snippet (first missing word context):**")
                            sample_token = tokens[0] if tokens else ""
                            snippet = get_pdf_snippet(stats['pdf_text'], sample_token)
                            st.text_area("", snippet, height=150, key=f"pdf_{sheet}_{ref}")
            
            # Full report text (for download)
            report = data['report']
            combined_report += f"\n{'='*70}\n{display_name.upper()} CHECKER REPORT\n{'='*70}\n{report}\n"
            st.text_area(f"{display_name} Full Report", report, height=200)
        
        # ---- 2.C: Download Excel Report ----
        if all_missing_cells:
            excel_data = generate_excel_report(results)
            st.download_button(
                label="⬇️ Download Excel Report (.xlsx)",
                data=excel_data,
                file_name="missing_words_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Also download text report
        st.download_button(
            label="⬇️ Download Full Combined Report (.txt)",
            data=combined_report,
            file_name="combined_missing_words_report.txt",
            mime="text/plain"
        )

st.markdown("---")
st.caption("Built with Streamlit | Processing runs on Google Colab backend.")
