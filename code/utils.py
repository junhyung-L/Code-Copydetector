"""This module contains functions for tokenizing/filtering code
as well as generic functions for detecting overlap between two
documents.
"""

import re
import logging
import warnings
from typing import Dict, List

from pygments import lexers, token
import pygments.util
import numpy as np
from markupsafe import escape

# if the C extention is available, use it. For almost all use cases
# the speed difference is not significant so if the C extention isn't
# found copydetect will silenty switch to the python implementation.
try:
    from .winnow import _winnow
except (ModuleNotFoundError, ImportError):
    from .pywinnow import _winnow

def filter_code(code, filename, language=None , include_comments: bool = False):
    """Tokenize and filter a code document. Replace variable names with
    V, function names with F, object names with O, and strings with S.
    Return the filtered document and a list of offsets indicating how
    many characters were removed by filtering at each index in the
    resulting document where filtering occured (this is used later to
    highlight the original code using plagiarism detection results on
    the filtered code)
    """
    try:
        lexer = lexers.get_lexer_by_name(language) if language else lexers.get_lexer_for_filename(filename)
        tokens = lexer.get_tokens(code)
        
    except Exception:
        logging.warning(f"{filename} not tokenized: unknown file extension")
        return code, np.array([])

    if lexer == pygments.lexers.TextLexer:
        logging.warning(f"did not tokenize plaintext file {filename}")
        return code, np.array([])

    out_code = ""
    offset = 0
    offsets = [[0, 0]]
    variable_tokens = {token.Name, token.Name.Variable, token.Name.Attribute}

    HANGUL = re.compile(r'[\uac00-\ud7a3]')
    for ttype, tval in tokens:
        if ttype in variable_tokens:
            out_code += "V"
            offsets.append([len(out_code) - 1, offset])
            offset += len(tval) - 1

        elif ttype in token.Name.Function:
            out_code += "F"
            offsets.append([len(out_code) - 1, offset])
            offset += len(tval) - 1

        elif ttype in token.Name.Class:
            out_code += "O"
            offsets.append([len(out_code) - 1, offset])
            offset += len(tval) - 1

        elif ttype == token.Comment.Preproc or ttype == token.Comment.Hashbang:
            # 전처리/해시뱅 주석도 동일 정책 적용
            if include_comments:
                out_code += tval
            else:
                offsets.append([len(out_code) - 1, offset])
                offset += len(tval)

        elif ttype in token.Comment:
            if include_comments:
                out_code += tval          #주석 포함
            else:
                offsets.append([len(out_code) - 1, offset])
                offset += len(tval)

        elif ttype in token.Text: 
            # 일부 케이스에서 한글 주석/문서형 텍스트가 Text 토큰으로 떨어짐
            # include_comments=True일 때, 한글이 하나라도 있으면 보존
            if include_comments and HANGUL.search(tval):
                out_code += tval
            else:
                offsets.append([len(out_code) - 1, offset])
                offset += len(tval)

        elif ttype in token.Literal.String:
            if tval == "'" or tval == '"':
                out_code += '"'
            else:
                out_code += "S"
                offsets.append([len(out_code) - 1, offset])
                offset += len(tval) - 1
        else:
            out_code += tval

    return out_code, np.array(offsets)

def hashed_kgrams(string, k):
    """Return hashes of all k-grams in a string"""
    hashes = [hash(string[offset:offset+k])
              for offset in range(len(string) - k + 1)]
    return np.array(hashes)

def winnow(hashes, window_size, remove_duplicates=True):
    """implementation of the robust winnowing algorithm decribed in
    https://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf
    Returns a list of selected hashes and the indexes of those hashes.
    """
    if window_size < 1:
        raise ValueError("window_size must be greater than 0")

    # window size of 1 will just select all hashes
    if window_size == 1:
        selected_hashes = hashes
        selected_idx = np.arange(len(hashes))
    else:
        selected_idx = _winnow(hashes, window_size)
        selected_hashes = hashes[selected_idx]

    if remove_duplicates:
        selected_hashes, unique_idx = np.unique(selected_hashes,
                                                return_index=True)
        selected_idx = selected_idx[unique_idx]

    return selected_hashes, selected_idx

def get_copied_slices(idx, k):
    """Given k and a list of indexes detected by
    find_fingerprint_overlap, generates a list of slices where the
    copied code begins and ends. Returns a 2D array where the first
    dimension is slice start locations and the second dimension is
    slice end locations.
    """
    if len(idx) == 0:
        return np.array([[],[]])

    # determine the gaps between slices (called skips)
    sorted_idx = np.sort(idx)
    next_idx = np.concatenate([sorted_idx[1:], [0]])
    skips = np.where(next_idx - sorted_idx > k - 1)[0]

    # use the elements around the gaps to compute slice start/ends
    slice_starts = np.concatenate([[sorted_idx[0]], sorted_idx[skips + 1]])
    slice_ends = np.concatenate([sorted_idx[skips]+k, [sorted_idx[-1]+k]])

    return np.array([slice_starts, slice_ends])

def get_document_fingerprints(doc, k, window_size, boilerplate=None):
    """Given a document, computes all k-gram hashes and uses the
    winnowing algorithm to reduce their number. Optionally takes a
    list of boilerplate hashes to remove from the winnowed list.
    Returns the selected hashes and their indexes in the original list
    """
    if boilerplate is None:
        boilerplate = []
    all_hashes = hashed_kgrams(doc, k=k)
    hashes, idx = winnow(
        all_hashes, window_size=window_size, remove_duplicates=False
    )
    if len(boilerplate) > 0:
        _, overlap_idx, _ = np.intersect1d(hashes, boilerplate,
                                           return_indices=True,
                                           assume_unique=True)
        idx = np.delete(idx, overlap_idx)
        hashes = np.delete(hashes, overlap_idx)

    hash_dict = {}
    for hash_val, i in zip(hashes, idx):
        if hash_val not in hash_dict:
            hash_dict[hash_val] = [i]
        else:
            hash_dict[hash_val].append(i)
    return set(hashes), hash_dict

def find_fingerprint_overlap(hashes1, hashes2, idx1, idx2):
    """Finds the indexes of overlapping values between two lists of
    hashes. Returns two lists of indexes, one for the first hash list
    and one for the second. The indexes of the original hashes are
    provided in case boilerplate results in gaps.
    """
    intersection = hashes1.intersection(hashes2)
    if len(intersection) > 0:
        overlap_1 = np.concatenate([np.array(idx1[i]) for i in intersection])
        overlap_2 = np.concatenate([np.array(idx2[i]) for i in intersection])
        return overlap_1.flatten(), overlap_2.flatten()
    else:
        return np.array([], dtype=int), np.array([], dtype=int)

def highlight_overlap(doc, slices, left_hl, right_hl,
                      truncate=-1, escape_html=False):
    """Highlights copied code in a document given the slices containing
    copied code and strings to use for the highlight start and end.
    Returns the document annoted with the highlight strings as well as
    the percentage of code which was highlighted. If truncate is set to
    an integer, everything not within that many lines of highlighted
    code will be replaced with "..."
    """
    if slices.shape[0] > 0:
        hl_percent = np.sum(slices[1] - slices[0])/len(doc)
    else:
        warnings.warn("empty slices array")
        return doc, 0

    new_doc = ""
    current_idx = 0
    for slice_idx in range(slices.shape[1]):
        start_idx = slices[0,slice_idx]
        end_idx = slices[1,slice_idx]

        if escape_html:
            pre_highlight = str(escape(doc[current_idx:start_idx]))
            highlighted = left_hl+str(escape(doc[start_idx:end_idx]))+right_hl
        else:
            pre_highlight = doc[current_idx:start_idx]
            highlighted = left_hl + doc[start_idx:end_idx] + right_hl

        if truncate >= 0:
            lines = pre_highlight.split("\n")
            if slice_idx != 0 and len(lines) > truncate*2:
                pre_highlight = ("\n".join(lines[:truncate+1]) + "\n\n...\n\n"
                                 + "\n".join(lines[-truncate - 1:]))
            elif len(lines) > truncate:
                pre_highlight = "\n".join(lines[-truncate - 1:])

        new_doc += pre_highlight + highlighted
        current_idx = end_idx

    if escape_html:
        post_highlight = str(escape(doc[current_idx:]))
    else:
        post_highlight = doc[current_idx:]

    if truncate >= 0:
        lines = post_highlight.split("\n")
        if len(lines) > truncate:
            post_highlight = "\n".join(lines[:truncate])
    new_doc += post_highlight

    return new_doc, hl_percent

def get_token_coverage(idx: Dict[int, List[int]], k: int, token_len: int):
    """Determines the number of tokens in the original document which
    are included in the winnowed indices
    """
    if len(idx) > 0:
        idx_arr = np.concatenate([np.array(i) for i in idx.values()])
    else:
        idx_arr = np.array([], dtype=int)
    coverage = np.zeros(token_len)
    for offset in range(k):
        coverage[idx_arr + offset] = 1
    return np.sum(coverage)

import re
from pygments import lexers, token

def extract_comments(code: str, filename: str, language: str | None = None) -> str:
    """코드에서 주석 텍스트만 추출. 
    일부 렉서가 한글 주석을 Text로 토큰화하는 케이스도 보완."""
    try:
        lexer = lexers.get_lexer_by_name(language) if language else lexers.get_lexer_for_filename(filename)
        tokens = lexer.get_tokens(code)
    except Exception:
        return ""

    HANGUL = re.compile(r'[\uac00-\ud7a3]')
    buf = []
    for ttype, tval in tokens:
        if ttype in token.Comment:
            buf.append(tval)
        elif ttype in token.Text and HANGUL.search(tval):
            buf.append(tval)
        elif ttype in token.Literal.String and HANGUL.search(tval):  # ★ 추가
            buf.append(tval)
    return "".join(buf)

# --- COMMENT-ONLY cosine similarity helpers ---
import math
from collections import Counter
import re

def _normalize_comment_text(s: str) -> str:
    # 한글/영문/숫자만 남기고 소문자, 공백 정규화
    s = re.sub(r'[^0-9A-Za-z\uac00-\ud7a3]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().lower()

def _char_ngrams(s: str, n: int = 3) -> Counter:
    if not s:
        return Counter()
    return Counter([s[i:i+n] for i in range(max(0, len(s)-n+1))])

def comment_cosine_sim(a: str, b: str, n: int = 3) -> float:
    a = _normalize_comment_text(a); b = _normalize_comment_text(b)
    if not a or not b:
        return 0.0
    ca, cb = _char_ngrams(a, n), _char_ngrams(b, n)
    keys = set(ca) | set(cb)
    dot = sum(ca[k]*cb.get(k, 0) for k in keys)
    na = math.sqrt(sum(v*v for v in ca.values()))
    nb = math.sqrt(sum(v*v for v in cb.values()))
    return (dot / (na*nb)) if (na and nb) else 0.0

# --- [ADD] utils.py ---
import nbformat

def load_ipynb_cells(path):
    """
    Returns:
      cells: [ {"type": "code"|"markdown", "source": str}, ... ]
    """
    nb = nbformat.read(path, as_version=4)
    cells = []
    for c in nb.cells:
        ctype = c.get("cell_type", "")
        src = c.get("source", "")
        cells.append({"type": ctype, "source": src})
    return cells

def ipynb_make_compare_text_and_map(path, include_markdown=False):
    """
    ipynb를 비교 가능한 문자열로 만들고, 전역오프셋→(cell_idx, line_start,line_end) 매핑을 만든다.
    - code 셀만 합침(기본). include_markdown=True면 마크다운도 포함.
    - 셀 경계에는 가시 마커를 삽입하되, 매핑에서 제외해 하이라이트 왜곡 방지.
    """
    cells = load_ipynb_cells(path)
    parts = []
    offset_map = []  # [(global_start_char, global_end_char, cell_idx, local_line_start, local_line_end)]
    gpos = 0
    for i, c in enumerate(cells):
        if c["type"] == "markdown" and not include_markdown:
            continue
        src = c["source"]
        if not src:
            continue
        # 줄 단위로 매핑
        lines = src.splitlines()
        # 셀 헤더(표시용): 매핑에는 포함시키지 않음
        header = f"# [cell {i} - {c['type']}]\n"
        parts.append(header)
        gpos += len(header)
        # 본문
        chunk = src if src.endswith("\n") else src + "\n"
        parts.append(chunk)
        # 라인 매핑 저장
        local_start = 1
        local_end = len(lines) if lines else 1
        offset_map.append((
            gpos, gpos + len(chunk), i, local_start, local_end
        ))
        gpos += len(chunk)
        # 셀 간 구분 공백
        sep = "\n"
        parts.append(sep)
        gpos += len(sep)
    compare_text = "".join(parts)
    return compare_text, offset_map, cells


