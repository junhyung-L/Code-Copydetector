"""This module contains functions for detecting overlap between
a set of test files (files to check for plagairism) and a set of
reference files (files that might have been plagairised from).
"""

from pathlib import Path
import time
import logging
import webbrowser
import importlib.resources
import io
import base64
import json
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from jinja2 import Template

import copydetect.data as data_files
from .utils import (filter_code, highlight_overlap, get_copied_slices,
                    get_document_fingerprints, find_fingerprint_overlap,
                    get_token_coverage, extract_comments,comment_cosine_sim)

from . import __version__
from . import defaults
from ._config import CopydetectConfig

class CodeFingerprint:
    """Class for tokenizing, filtering, fingerprinting, and winnowing
    a file. Maintains information about fingerprint indexes and token
    indexes to assist code highlighting for the output report.

    Parameters
    ----------
    file : str
        Path to the file fingerprints should be extracted from.
    k : int
        Length of k-grams to extract as fingerprints.
    win_size : int
        Window size to use for winnowing (must be >= 1).
    boilerplate : array_like, optional
        List of fingerprints to use as boilerplate. Any fingerprints
        present in this list will be discarded from the hash list.
    filter : bool, default=True
        If set to to False, code will not be tokenized & filtered.
    fp : TextIO, default=None
        I/O stream for data to create a fingerprint for. If provided,
        the "file" argument will not be used to load a file from disk
        but will still be used for language detection and displayed on
        the report.
    encoding : str, default="utf-8"
        Text encoding to use for reading the file. If "DETECT", the
        chardet library will be used (if installed) to automatically
        detect file encoding

    Attributes
    ----------
    filename : str
        Name of the originally provided file.
    raw_code : str
        Unfiltered code.
    filtered_code : str
        Code after tokenization and filtering. If filter=False, this is
        the same as raw_code.
    offsets : Nx2 array of ints
        The cumulative number of characters removed during filtering at
        each index of the filtered code. Used for translating locations
        in the filtered code to locations in the unfiltered code.
    hashes : Set[int]
        Set of fingerprint hashes extracted from the filtered code.
    hash_idx : Dict[int, List[int]]
        Mapping of each fingerprint hash back to all indexes in the
        original code in which this fingerprint appeared.
    k : int
        Value of provided k argument.
    language : str
        If set, will force the tokenizer to use the provided language
        rather than guessing from the file extension.
    token_coverage : int
        The number of tokens in the tokenized code which are considered
        for fingerprint comparison, after performing winnowing and
        removing boilerplate.
    """
    def __init__(self, file, k, win_size, boilerplate=None, filter=True,
                 language=None, fp=None, encoding: str = "utf-8"):
        if boilerplate is None:
            boilerplate = []
        if fp is not None:
            code = fp.read()
        elif encoding == "DETECT":
            try:
                import chardet
                with open(file, "rb") as code_fp:
                    code = code_fp.read()
                detected_encoding = chardet.detect(code)["encoding"]
                if detected_encoding is not None:
                    code = code.decode(detected_encoding)
                else:
                    # if encoding can't be detected, just use the default
                    # encoding (the file may be empty)
                    code = code.decode()
            except ModuleNotFoundError as e:
                logging.error(
                    "encoding detection requires chardet to be installed"
                )
                raise e
        else:
            with open(file, encoding=encoding) as code_fp:
                code = code_fp.read()
        if filter:
            filtered_code, offsets = filter_code(code, file, language, include_comments=False)
        else:
            filtered_code, offsets = code, np.array([])
        hashes, idx = get_document_fingerprints(filtered_code, k, win_size,
                                                boilerplate)

        self.filename = file
        self.raw_code = code
        self.filtered_code = filtered_code
        self.offsets = offsets
        self.hashes = hashes
        self.hash_idx = idx
        self.k = k
        self.token_coverage = get_token_coverage(idx, k, len(filtered_code))

        # comment-only fingerprints (HTML 분리 섹션용, 선택적)
        self.comment_text = ""
        self.comment_hashes = set()
        self.comment_hash_idx = {}



def compare_files(file1_data, file2_data):
    """Computes the overlap between two CodeFingerprint objects
    using the generic methods from copy_detect.py. Returns the
    number of overlapping tokens and two tuples containing the
    overlap percentage and copied slices for each unfiltered file.

    Parameters
    ----------
    file1_data : CodeFingerprint
        CodeFingerprint object of file #1.
    file2_data : CodeFingerprint
        CodeFingerprint object of file #2.

    Returns
    -------
    token_overlap : int
        Number of overlapping tokens between the two files.
    similarities : tuple of 2 ints
        For both files: number of overlapping tokens divided by the
        total number of tokens in that file.
    slices : tuple of 2 2xN int arrays
        For both files: locations of copied code in the unfiltered
        text. Dimension 0 contains slice starts, dimension 1 contains
        slice ends.
    """
    if file1_data.k != file2_data.k:
        raise ValueError("Code fingerprints must use the same noise threshold")
    idx1, idx2 = find_fingerprint_overlap(
        file1_data.hashes, file2_data.hashes,
        file1_data.hash_idx, file2_data.hash_idx)
    slices1 = get_copied_slices(idx1, file1_data.k)
    slices2 = get_copied_slices(idx2, file2_data.k)
    if len(slices1[0]) == 0:
        return 0, (0,0), (np.array([]), np.array([]))

    token_overlap1 = np.sum(slices1[1] - slices1[0])
    token_overlap2 = np.sum(slices2[1] - slices2[0])

    if len(file1_data.filtered_code) > 0:
        similarity1 = token_overlap1 / file1_data.token_coverage
    else:
        similarity1 = 0
    if len(file2_data.filtered_code) > 0:
        similarity2 = token_overlap2 / file2_data.token_coverage
    else:
        similarity2 = 0

    if len(file1_data.offsets) > 0:
        slices1 += file1_data.offsets[:,1][np.clip(
            np.searchsorted(file1_data.offsets[:,0], slices1),
            0, file1_data.offsets.shape[0] - 1)]
    if len(file2_data.offsets) > 0:
        slices2 += file2_data.offsets[:,1][np.clip(
            np.searchsorted(file2_data.offsets[:,0], slices2),
            0, file2_data.offsets.shape[0] - 1)]

    return token_overlap1, (similarity1,similarity2), (slices1,slices2)

from jinja2 import Template
import os
import shutil
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup  # pip install beautifulsoup4

class CopyDetector:
    """Main plagairism detection class. Searches provided directories
    and uses detection parameters to calculate similarity between all
    files found in the directories

    Parameters
    ----------
    test_dirs : list
        (test_directories) A list of directories to recursively search
        for files to check for plagiarism.
    ref_dirs: list
        (reference_directories) A list of directories to search for
        files to compare the test files to. This should generally be a
        superset of test_directories
    boilerplate_dirs : list
        (boilerplate_directories) A list of directories containing
        boilerplate code. Matches between fingerprints present in the
        boilerplate code will not be considered plagiarism.
    extensions : list
        A list of file extensions containing code the detector should
        look at.
    noise_t : int
        (noise_threshold) The smallest sequence of matching characters
        between two files which should be considered plagiarism. Note
        that tokenization and filtering replaces variable names with V,
        function names with F, object names with O, and strings with S
        so the threshold should be lower than you would expect from the
        original code.
    guarantee_t : int
        (guarantee_threshold) The smallest sequence of matching
        characters between two files for which the system is guaranteed
        to detect a match. This must be greater than or equal to the
        noise threshold. If computation time is not an issue, you can
        set guarantee_threshold = noise_threshold.
    display_t : float
        (display_threshold) The similarity percentage cutoff for
        displaying similar files on the detector report.
    same_name_only : bool
        If true, the detector will only compare files that have the
        same name
    ignore_leaf : bool
        If true, the detector will not compare files located in the
        same leaf directory.
    autoopen : bool
        If true, the detector will automatically open a webbrowser to
        display the results of generate_html_report
    disable_filtering : bool
        If true, the detector will not tokenize and filter code before
        generating file fingerprints.
    force_language : str
        If set, forces the tokenizer to use a particular programming
        language regardless of the file extension.
    truncate : bool
        If true, highlighted code will be truncated to remove non-
        highlighted regions from the displayed output
    out_file : str
        Path to output report file.
    css_files: list
        List of css files that will be linked within the generated html report
    silent : bool
        If true, all logging output will be supressed.
    encoding : str, default="utf-8"
        Text encoding to use for reading the file. If "DETECT", the
        chardet library will be used (if installed) to automatically
        detect file encoding
    """
    def __init__(self, test_dirs=None, ref_dirs=None,
                 boilerplate_dirs=None, extensions=None,
                 noise_t=defaults.NOISE_THRESHOLD,
                 guarantee_t=defaults.GUARANTEE_THRESHOLD,
                 display_t=defaults.DISPLAY_THRESHOLD,
                 same_name_only=False, ignore_leaf=False, autoopen=True,
                 disable_filtering=False, force_language=None,
                 truncate=False, out_file="./report.html", css_files=None,
                 silent=False, encoding: str = "utf-8",
                 html_separate_comment_score: bool = False,
                 comment_noise_t: int = 10,
                 comment_guarantee_t: int = 10,
                 comment_mode: str = "kgram",
                 comment_ngram_n: int = 3,
                 comment_weight: float = 1.0,
                 include_markdown: bool = False,
                 include_raw: bool = False,
                 html_notebook_preview: bool = False,        # ★ 추가
                 notebook_preview_dir: str = "notebook_preview"):
        conf_args = locals()
        conf_args = {k: v for k, v in conf_args.items() if k != "self" and v is not None}
        self.conf = CopydetectConfig(**conf_args)

        self.test_files = self._get_file_list(
            self.conf.test_dirs, self.conf.extensions
        )
        self.ref_files = self._get_file_list(
            self.conf.ref_dirs, self.conf.extensions
        )
        self.boilerplate_files = self._get_file_list(
            self.conf.boilerplate_dirs, self.conf.extensions
        )

        # before run() is called, similarity data should be empty
        self.similarity_matrix = np.array([])
        self.token_overlap_matrix = np.array([])
        self.slice_matrix = {}
        self.file_data = {}
        self._ipynb_cell_spans = {}     # {filepath: [(start,end,cell_idx), ...] in concatenated code}
        self._ipynb_cell_srcs  = {}
        self._ipynb_render_base = {}    # {filepath: rendered_html_path (no highlight)}
        self._pair_preview = []         # [(test_f, ref_f, test_html, ref_html)]


    @classmethod
    def from_config(cls, config):
        """Initializes a CopyDetection object using the provided
        configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary using CLI parameter names.

        Returns
        -------
        CopyDetector
            Detection object initialized with config
        """
        params = CopydetectConfig.normalize_json(config)
        return cls(**params)

    def _get_file_list(self, dirs, exts):
        """Recursively collects list of files from provided
        directories. Used to search test_dirs, ref_dirs, and
        boilerplate_dirs
        """
        file_list = []
        for dir in dirs:
            print_warning = True
            for ext in exts:
                if ext == "*":
                    matched_contents = Path(dir).rglob("*")
                else:
                    matched_contents = Path(dir).rglob("*."+ext.lstrip("."))
                files = [str(f) for f in matched_contents if f.is_file()]

                if len(files) > 0:
                    print_warning = False
                file_list.extend(files)
            if print_warning:
                logging.warning("No files found in " + dir)

        # convert to a set to remove duplicates, then back to a list
        return list(set(file_list))

    def add_file(self, filename, type="testref"):
        """Adds a file to the list of test files, reference files, or
        boilerplate files.

        Parameters
        ----------
        filename : str
            Name of file to add.
        type : {"testref", "test", "ref", "boilerplate"}
            Type of file to add. "testref" will add the file as both a
            test and reference file.
        """
        if type == "testref":
            self.test_files.append(filename)
            self.ref_files.append(filename)
        elif type == "test":
            self.test_files.append(filename)
        elif type == "ref":
            self.ref_files.append(filename)
        elif type == "boilerplate":
            self.boilerplate_files.append(filename)

    def _get_boilerplate_hashes(self):
        """Generates a list of hashes of the boilerplate text. Returns
        a set containing all unique k-gram hashes across all files
        found in the boilerplate directories.
        """
        boilerplate_hashes = []
        for file in self.boilerplate_files:
            try:
                fingerprint = CodeFingerprint(
                    file,
                    k=self.conf.noise_t,
                    win_size=1,
                    filter=not self.conf.disable_filtering,
                    language=self.conf.force_language,
                    encoding=self.conf.encoding
                )
                boilerplate_hashes.extend(fingerprint.hashes)
            except UnicodeDecodeError:
                logging.warning(f"Skipping {file}: file not UTF-8 text")
                continue

        return np.unique(np.array(boilerplate_hashes))

    def _preprocess_code(self, file_list):
        """Generates a CodeFingerprint object for each file in the
        provided file list. This is where the winnowing algorithm is
        actually used.
        """
        boilerplate_hashes = self._get_boilerplate_hashes()
        for code_f in tqdm(file_list, bar_format= '   {l_bar}{bar}{r_bar}',
                           disable=self.conf.silent):
                
            if code_f not in self.file_data:
                if code_f.endswith(".ipynb"):
                    code_text, cell_spans, cell_srcs = self._extract_ipynb_code(code_f)
                    self._ipynb_cell_spans[code_f] = cell_spans
                    self._ipynb_cell_srcs[code_f]  = cell_srcs

                    # === [ADD] markdown/raw 포함 시 필터 OFF ===
                    want_text = (self.conf.include_markdown or self.conf.include_raw)
                    use_filter = (not want_text) and (not self.conf.disable_filtering)
                    
                    self.file_data[code_f] = CodeFingerprint(
                        code_f,
                        k=self.conf.noise_t,
                        win_size=self.conf.guarantee_t,  # ★ window_size 필드가 없다면 guarantee_t 사용
                        boilerplate=boilerplate_hashes,
                        filter=use_filter,
                        language="python",
                        fp=io.StringIO(code_text),
                        encoding=self.conf.encoding
                    )
                    if self.conf.html_notebook_preview:
                        base_html = self._render_ipynb_html(code_f)
                        if base_html:
                            self._ipynb_render_base[code_f] = base_html
                else:
                    self.file_data[code_f] = CodeFingerprint(
                        code_f,
                        k=self.conf.noise_t,
                        win_size=self.conf.guarantee_t,  # ★ 같은 이유로 guarantee_t 추천
                        boilerplate=boilerplate_hashes,
                        filter=not self.conf.disable_filtering,
                        language=self.conf.force_language,
                        encoding=self.conf.encoding
                    )

    # --- ipynb helpers ---
    def _extract_ipynb_code(self, ipynb_path: str):
        """ipynb에서 코드셀만 이어붙인 텍스트와, 각 셀의 전역 오프셋 구간 목록을 반환"""
        nb = nbformat.read(ipynb_path, as_version=4)

        # === [CHANGE] 포함할 셀 타입 결정 ===
        target_types = {"code"}
        if getattr(self.conf, "include_markdown", False):
            target_types.add("markdown")
        if getattr(self.conf, "include_raw", False):
            target_types.add("raw")
                
        parts, spans, cell_srcs = [], [], []
        cur = 0
        
        for ci, c in enumerate(nb["cells"]):
            if c.get("cell_type") not in target_types:
                continue
            src = c.get("source", "")
            s = src if src.endswith("\n") else src + "\n"
            parts.append(s)
            start, end = cur, cur + len(s)
            spans.append((start, end, ci))
            cell_srcs.append(src)
            cur = end

        return ("".join(parts), spans, cell_srcs)

    def _render_ipynb_html(self, ipynb_path: str) -> str | None:
        """nbconvert로 정적 HTML 렌더링(베이스)."""
        try:
            export = HTMLExporter()
            export.exclude_output_prompt = True
            body, _ = export.from_filename(ipynb_path)
            out_dir = Path(self.conf.notebook_preview_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / (Path(ipynb_path).stem + "_base.html")
            dst.write_text(body, encoding="utf-8")
            return str(dst)
        except Exception as e:
            logging.warning(f"nbconvert failed for {ipynb_path}: {e}")
            return None

    def _sym_score(self, s_ab: float, s_ba: float, mode: str = "dice") -> float:
        """
        대칭(symmetric) 표절률 생성. 모드는 dice|jaccard|max|mean 중 택1.
        - dice(F1): 2ab/(a+b)  (추천: 직관적)
        - jaccard: ab / (a + b - ab)  (보수적)
        - max: max(a,b)  (짧은 문서 포함관계에 민감)
        - mean: (a+b)/2  (간단 평균)
        """
        a, b = float(s_ab), float(s_ba)
        if a <= 0 and b <= 0:
            return 0.0
        if mode == "dice":
            return (2*a*b)/(a+b) if (a+b) > 0 else 0.0
        if mode == "jaccard":
            den = a + b - (a*b)
            return (a*b)/den if den > 0 else 0.0
        if mode == "max":
            return max(a, b)
        if mode == "mean":
            return (a + b) / 2.0
        return (2*a*b)/(a+b) if (a+b) > 0 else 0.0


    def _make_pair_highlighted_notebook(self, base_html: str, file_path: str, slices, color_class: str) -> str | None:
        """
        페어(한 쌍) 기준으로 하이라이트 반영한 노트북 HTML 생성.
        slices: 2xN array (start,end) - 이 파일 입장에서의 전역 문자열 오프셋
        """
        if file_path not in self._ipynb_cell_spans:
            return None
        spans = self._ipynb_cell_spans[file_path]  # [(s,e,cell_idx), ...]
        try:
            html = Path(base_html).read_text(encoding="utf-8")
            soup = BeautifulSoup(html, "html.parser")
            
            # 노트북 HTML의 코드셀 <pre>들을 순서대로 수집
            # 1) 코드셀 <pre>들을 폭넓게 수집(템플릿 차이 흡수)  :contentReference[oaicite:1]{index=1}
            pre_blocks = soup.select(
            "div.input_area pre, "
            "div.input_area div.highlight pre, "
            "div.jp-InputArea-editor pre, "
            "div.cell.code div.input_area pre, "
            "div.nbinput div.input_area pre"
            )
            # 중복 제거(있을 경우)
            seen, uniq = set(), []
            for el in pre_blocks:
                if id(el) not in seen:
                    seen.add(id(el)); uniq.append(el)
            pre_blocks = uniq

            # 2) 전역 오프셋 구간을 '셀 로컬 구간'으로 모으기 (ci -> [(ls, le), ...])
            from collections import defaultdict
            cell_spans_map = defaultdict(list)
            for (start_g, end_g) in zip(slices[0], slices[1]):
                for (cs, ce, ci) in spans:  # 셀의 전역 범위는 _extract_ipynb_code에서 만든 것 사용 :contentReference[oaicite:2]{index=2}
                    if end_g <= cs or start_g >= ce:
                        continue
                    ls = max(0, start_g - cs)
                    le = min(ce - cs, end_g - cs)
                    if le > ls:
                        cell_spans_map[ci].append((ls, le))

            # 3) 구간 병합(겹치거나 맞닿으면 합침)
            def merge_intervals(iv):
                if not iv:
                    return []
                iv = sorted(iv, key=lambda x: (x[0], x[1]))
                merged = []
                s, e = iv[0]
                for a, b in iv[1:]:
                    if a <= e:
                        e = max(e, b)
                    else:
                        merged.append((s, e))
                        s, e = a, b
                merged.append((s, e))
                return merged

            # 4) 각 셀을 '원문으로 초기화'하고 원문 기준으로 하이라이트 삽입
            cell_srcs = self._ipynb_cell_srcs.get(file_path, [])
            for ci, iv in cell_spans_map.items():
                if ci >= len(pre_blocks) or ci >= len(cell_srcs):
                    continue

            # (a) 이 셀의 원문을 기준 문자열로. 오프셋과 일치시키기 위해 끝 개행을 보정
            src0 = cell_srcs[ci]
            base = src0 if src0.endswith("\n") else (src0 + "\n")

            # (b) 병합+역순 삽입(인덱스 보존)
            intervals = merge_intervals(iv)
            intervals.sort(key=lambda x: (x[0], x[1]), reverse=True)
            out = base
            for ls, le in intervals:
                # 안전 클램프
                ls = max(0, min(ls, len(out)))
                le = max(ls, min(le, len(out)))
                out = out[:ls] + "{{HL_START}}" + out[ls:le] + "{{HL_END}}" + out[le:]

            # (c) 원문이 개행 없이 끝났다면, 시각 효과 유지 위해 마지막 \n 제거
            if not src0.endswith("\n") and out.endswith("\n"):
                out = out[:-1]

            # (d) DOM 재구성: 텍스트를 마커 기준으로 쪼개고 <span class=color_class>로 감쌈
            pre = pre_blocks[ci]
            pre.clear()
            parts = out.split("{{HL_START}}")
            pre.append(parts[0])
            for part in parts[1:]:
                frag, rest = part.split("{{HL_END}}", 1)
                span = soup.new_tag("span", **{"class": color_class})
                span.string = frag
                pre.append(span)
                pre.append(rest)

            # 5) 스타일 주입(보고서와 동일 클래스)  :contentReference[oaicite:3]{index=3}
            style = soup.new_tag("style")
            style.string = """
            .highlight-red { background: rgba(255, 0, 0, 0.25); }
            .highlight-green { background: rgba(0, 255, 0, 0.25); }
            """
            soup.head.append(style)

            # 6) 출력 보장: 디렉터리 생성 후 저장
            out_dir = Path(self.conf.notebook_preview_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pair_name = f"{Path(file_path).stem}_{hash(tuple(map(int, slices.flatten())))}.html"
            out_path = out_dir / pair_name
            out_path.write_text(str(soup), encoding="utf-8")
            return str(out_path)

        except Exception as e:
            logging.warning(f"inject highlights failed for {file_path}: {e}")
            return None
                
                # 별도: HTML 분리 스코어가 켜진 경우에만 주석 지문 생성
            if self.conf.html_separate_comment_score:
                f = self.file_data[code_f]
                f.comment_text = extract_comments(f.raw_code, code_f, self.conf.force_language)
                if len(f.comment_text) > 0:
                    f.k_comment = self.conf.comment_noise_t
                    f.win_comment = self.conf.window_size_comment
                    f.comment_hashes, f.comment_hash_idx = get_document_fingerprints(
                        f.comment_text, f.k_comment, f.win_comment, boilerplate=None
                    )
                        
    def _comparison_loop(self):
        """The core code used to determine code overlap. The overlap
        between each test file and each compare file is computed and
        stored in similarity_matrix. Token overlap information and the
        locations of copied code are stored in slice_matrix and
        token_overlap_matrix, respectively.
        """

        self.similarity_matrix = np.full(
            (len(self.test_files), len(self.ref_files), 2),
            -1,
            dtype=np.float64,
        )
        self.token_overlap_matrix = np.full(
            (len(self.test_files), len(self.ref_files)), -1
        )
        self.slice_matrix = {}

        # this is used to track which files have been compared to avoid
        # unnecessary duplication when there is overlap between the
        # test and reference files
        comparisons = {}

        for i, test_f in enumerate(
            tqdm(self.test_files,
                 bar_format= '   {l_bar}{bar}{r_bar}',
                 disable=self.conf.silent)
        ):
            for j, ref_f in enumerate(self.ref_files):
                if (test_f not in self.file_data
                        or ref_f not in self.file_data
                        or test_f == ref_f
                        or (self.conf.same_name_only
                            and (Path(test_f).name != Path(ref_f).name))
                        or (self.conf.ignore_leaf
                            and (Path(test_f).parent == Path(ref_f).parent))):
                    continue

                if (ref_f, test_f) in comparisons:
                    ref_idx, test_idx = comparisons[(ref_f, test_f)]
                    overlap = self.token_overlap_matrix[ref_idx, test_idx]
                    sim2, sim1 = self.similarity_matrix[ref_idx, test_idx]
                else:
                     overlap, sims, slices = compare_files(
                        self.file_data[test_f], self.file_data[ref_f]
                )
                sim1, sim2 = sims
                self.token_overlap_matrix[i, j] = overlap

                '''

                # === 주석 전용 유사도(코사인 or k-gram) 포함해 최종 합산 ===
                if self.conf.html_separate_comment_score:
                    f1, f2 = self.file_data[test_f], self.file_data[ref_f]

                    # 전체 분모 = 코드길이 + 주석길이 (합산 철학은 그대로 유지)
                    den1 = max(1, len(f1.filtered_code) + len(getattr(f1, "comment_text", "")))
                    den2 = max(1, len(f2.filtered_code) + len(getattr(f2, "comment_text", "")))

                    if getattr(self.conf, "comment_mode", "kgram") == "cosine":
                        # --- 코사인 모드: 주석 텍스트로 0~1 스코어 산출 ---
                        c1 = getattr(f1, "comment_text", "")
                        c2 = getattr(f2, "comment_text", "")
                        if c1 and c2:
                            # 너무 짧은 주석을 위한 n 보정(최소 1)
                            n = getattr(self.conf, "comment_ngram_n", 3)
                            n = max(1, min(n, len(c1), len(c2)))
                            s = comment_cosine_sim(c1, c2, n=n)  # 0~1
                            w = getattr(self.conf, "comment_weight", 1.0)

                            # 길이 가중 등가 overlap으로 환산해서 합산
                            ovl1 = int(s * len(c1) * w)
                            ovl2 = int(s * len(c2) * w)
                            sim1 = (sim1 * len(f1.filtered_code) + ovl1) / den1
                            sim2 = (sim2 * len(f2.filtered_code) + ovl2) / den2
                            overlap = overlap + max(ovl1, ovl2)

                    else:
                        # --- 기존 k-gram 주석 fingerprint 경로 유지 ---
                        if len(getattr(f1, "comment_hashes", [])) > 0 and len(getattr(f2, "comment_hashes", [])) > 0:
                            oc1, oc2 = find_fingerprint_overlap(
                                f1.comment_hashes, f2.comment_hashes,
                                f1.comment_hash_idx, f2.comment_hash_idx
                            )
                            s1 = get_copied_slices(oc1, f1.k_comment)
                            s2 = get_copied_slices(oc2, f2.k_comment)
                            ovl1 = int(np.sum(s1[1] - s1[0])) if s1.shape[0] > 0 else 0
                            ovl2 = int(np.sum(s2[1] - s2[0])) if s2.shape[0] > 0 else 0
                            sim1 = (sim1 * len(f1.filtered_code) + ovl1) / den1
                            sim2 = (sim2 * len(f2.filtered_code) + ovl2) / den2
                            overlap = overlap + max(ovl1, ovl2)

                '''
                # === 주석 전용 유사도(코사인 or k-gram) 포함해 최종 합산 ===
                if self.conf.html_separate_comment_score:
                    f1, f2 = self.file_data[test_f], self.file_data[ref_f]

                    # (핵심) 희석 방지: 분모는 '코드 길이'만 사용
                    den1 = max(1, len(f1.filtered_code))
                    den2 = max(1, len(f2.filtered_code))

                    mode = getattr(self.conf, "comment_mode", "kgram")
                    if mode == "cosine":
                        c1 = getattr(f1, "comment_text", "")
                        c2 = getattr(f2, "comment_text", "")
                        if c1 and c2:
                            n = max(1, min(getattr(self.conf, "comment_ngram_n", 3), len(c1), len(c2)))
                            s = comment_cosine_sim(c1, c2, n=n)     # 0~1
                            w = getattr(self.conf, "comment_weight", 1.0)

                            # (핵심) 내림 제거: 부동으로 '가산만'
                            ovl1 = s * len(c1) * w
                            ovl2 = s * len(c2) * w
                            sim1 = (sim1 * den1 + ovl1) / den1
                            sim2 = (sim2 * den2 + ovl2) / den2

                            # (시각화) 코사인은 위치가 없으니, 주석 k-gram 슬라이스를 코드 슬라이스에 합쳐서 칠한다
                            if len(getattr(f1, "comment_hashes", [])) > 0 and len(getattr(f2, "comment_hashes", [])) > 0:
                                oc1, oc2 = find_fingerprint_overlap(
                                    f1.comment_hashes, f2.comment_hashes,
                                    f1.comment_hash_idx, f2.comment_hash_idx
                                )
                                s1c = get_copied_slices(oc1, f1.k_comment)
                                s2c = get_copied_slices(oc2, f2.k_comment)
                                import numpy as _np
                                # slices는 (test_slices, ref_slices) 튜플 → 각각에 합쳐준다
                                if isinstance(slices, tuple) and len(slices) == 2:
                                    st, sr = slices
                                    if getattr(s1c, "size", 0):
                                        st = (_np.hstack([st, s1c]) if getattr(st, "size", 0) else s1c)
                                    if getattr(s2c, "size", 0):
                                        sr = (_np.hstack([sr, s2c]) if getattr(sr, "size", 0) else s2c)
                                    # 시작 위치 기준 정렬(보기 좋게)
                                    if getattr(st, "size", 0):
                                        st = st[:, _np.argsort(st[0])]
                                    if getattr(sr, "size", 0):
                                        sr = sr[:, _np.argsort(sr[0])]
                                    slices = (st, sr)

                    else:
                        # (선택) k-gram 주석 경로 유지하고 싶을 때
                        if len(getattr(f1, "comment_hashes", [])) > 0 and len(getattr(f2, "comment_hashes", [])) > 0:
                            oc1, oc2 = find_fingerprint_overlap(
                                f1.comment_hashes, f2.comment_hashes,
                                f1.comment_hash_idx, f2.comment_hash_idx
                            )
                            s1 = get_copied_slices(oc1, f1.k_comment)
                            s2 = get_copied_slices(oc2, f2.k_comment)
                            ovl1 = float(np.sum(s1[1] - s1[0])) if s1.shape[0] > 0 else 0.0
                            ovl2 = float(np.sum(s2[1] - s2[0])) if s2.shape[0] > 0 else 0.0
                            sim1 = (sim1 * den1 + ovl1) / den1
                            sim2 = (sim2 * den2 + ovl2) / den2
                            # 색칠: 주석 슬라이스도 합치기
                            import numpy as _np
                            st, sr = slices
                            if getattr(s1, "size", 0):
                                st = (_np.hstack([st, s1]) if getattr(st, "size", 0) else s1)
                            if getattr(s2, "size", 0):
                                sr = (_np.hstack([sr, s2]) if getattr(sr, "size", 0) else s2)
                            slices = (st, sr)

                sym = max(sim1, sim2)  # 두 방향 중 큰 값

                self.similarity_matrix[i, j, 0] = sym
                self.similarity_matrix[i, j, 1] = sym
                self.slice_matrix[(test_f, ref_f)] = slices
                # (선택) 역방향 키도 같은 슬라이스로 채워두면 조회가 편함
                self.slice_matrix[(ref_f, test_f)] = (slices[1], slices[0])

    def run(self):
        """Runs the copy detection loop for detecting overlap between
        test and reference files. If no files are in the provided
        directories, the similarity matrix will remain empty and any
        attempts to generate a report will fail.
        """
        if len(self.test_files) == 0:
            logging.error("Copy detector failed: No files found in "
                          "test directories")
        elif len(self.ref_files) == 0:
            logging.error("Copy detector failed: No files found in "
                          "reference directories")
        else:
            start_time = time.time()

            if not self.conf.silent:
                print("  0.00: Generating file fingerprints")

            self._preprocess_code(self.test_files + self.ref_files)

            if not self.conf.silent:
                print(f"{time.time()-start_time:6.2f}: Beginning code comparison")

            self._comparison_loop()

            if not self.conf.silent:
                print(f"{time.time()-start_time:6.2f}: Code comparison completed")

    def get_copied_code_list(self):
        """Matched Code 섹션에 쓸 데이터 생성 (ipynb면 좌/우 iframe URL도 함께 제공)"""
        if len(self.similarity_matrix) == 0:
            logging.error("Cannot generate code list: no files compared")
            return []

        s_ab = self.similarity_matrix[:, :, 0]
        s_ba = self.similarity_matrix[:, :, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            sym_mtx = (2.0 * s_ab * s_ba) / (s_ab + s_ba)
        # ★ 대칭 점수 기준으로 페어 선택
        x, y = np.where(sym_mtx > self.conf.display_t)

        code_list = []
        file_pairs = set()

        def _normalize_slices(slices):
            """(2, N) 슬라이스 배열을 시작좌표 기준으로 정렬하고 겹침 병합."""
            import numpy as np
            if getattr(slices, "size", 0) == 0 or slices.shape[1] == 0:
                return slices
            # 시작 위치 기준 정렬
            order = np.argsort(slices[0])
            starts = slices[0][order]
            ends   = slices[1][order]

            merged_starts = []
            merged_ends = []
            cur_s, cur_e = int(starts[0]), int(ends[0])
            for s, e in zip(starts[1:], ends[1:]):
                s, e = int(s), int(e)
                if s <= cur_e:          # 겹치거나 접하면 병합
                    cur_e = max(cur_e, e)
                else:
                    merged_starts.append(cur_s); merged_ends.append(cur_e)
                    cur_s, cur_e = s, e
            merged_starts.append(cur_s); merged_ends.append(cur_e)

            return np.vstack([merged_starts, merged_ends]).astype(int)

        for idx in range(len(x)):
            ti, ri = int(x[idx]), int(y[idx])
            test_f = self.test_files[ti]
            ref_f  = self.ref_files[ri]
            if (ref_f, test_f) in file_pairs:
                continue
            file_pairs.add((test_f, ref_f))

            # --- 변경(대칭 점수 계산) ---
            s_ab = float(self.similarity_matrix[ti, ri, 0])
            s_ba = float(self.similarity_matrix[ti, ri, 1])
            # 원하는 방식 선택: max / mean / dice
            #sym = max(s_ab, s_ba)
            # sym = (s_ab + s_ba) / 2
            sym = (2*s_ab*s_ba)/(s_ab+s_ba) if (s_ab+s_ba)>0 else 0.0

            # --- slices 안전하게 확보 (없으면 빈 (2,0) 배열) ---
            import numpy as _np
            def _empty_slices():
                return _np.array([[],[]], dtype=int)

            pair = self.slice_matrix.get((test_f, ref_f))
            if pair is not None:
                slices_test, slices_ref = pair
            else:
                # 역방향 키에 저장된 경우
                pair_rev = self.slice_matrix.get((ref_f, test_f))
                if pair_rev is not None:
                    slices_ref, slices_test = pair_rev
                else:
                    slices_test, slices_ref = _empty_slices(), _empty_slices()

            slices_test = _normalize_slices(slices_test)
            slices_ref  = _normalize_slices(slices_ref)

            # slices -> spans(dict)로 변환
            def _to_spans(slices, score=1.0):
                if getattr(slices, "size", 0) == 0 or slices.shape[1] == 0:
                    return []
                return [
                {"start": int(s), "end": int(e), "score": float(score)}
                for s, e in zip(slices[0], slices[1])
            ]

            spans_test = _to_spans(slices_test, score=sym)   # sym = Dice 점수(이미 위에서 계산)
            spans_ref  = _to_spans(slices_ref,  score=sym)

            # 안전 렌더러로 하이라이트 HTML 생성 (라인 분할 없음, 오른->왼 삽입, 병합/클램프 내장)
            hl_code_1 = self._render_highlighted_html(self.file_data[test_f].raw_code, spans_test)
            hl_code_2 = self._render_highlighted_html(self.file_data[ref_f].raw_code,  spans_ref)

            # overlap (없으면 -1일 수 있음)
            overlap = -1
            if self.token_overlap_matrix.size:
                overlap = int(self.token_overlap_matrix[ti, ri])

            # ★ ipynb면 좌/우 iframe으로 보여줄 수 있도록 쌍별 HTML 생성 + 상대 URL 제공
            ipynb_left_url  = None
            ipynb_right_url = None
            if self.conf.html_notebook_preview and test_f.endswith(".ipynb") and ref_f.endswith(".ipynb"):
                base_test = self._ipynb_render_base.get(test_f)
                base_ref  = self._ipynb_render_base.get(ref_f)
                if base_test and base_ref:
                    test_html = self._make_pair_highlighted_notebook(base_test, test_f, slices_test, "highlight-red")
                    ref_html  = self._make_pair_highlighted_notebook(base_ref,  ref_f,  slices_ref,  "highlight-green")
                    if test_html and ref_html:
                        ipynb_left_url  = self._to_rel_url(test_html)
                        ipynb_right_url = self._to_rel_url(ref_html)

            # 최종 레코드: test_sim, ref_sim 대신 sym을 두 번 넣어 통일
            code_list.append([
                sym, sym, test_f, ref_f,
                hl_code_1, hl_code_2, overlap,
                ipynb_left_url, ipynb_right_url
            ])

        code_list.sort(key=lambda row: -row[0])
        return code_list


    def generate_html_report(self, output_mode="save"):
        """Generates an html report listing all files with similarity
        above the display_threshold, with the copied code segments
        highlighted.

        Parameters
        ----------
        output_mode : {"save", "return"}
            If "save", the output will be saved to the file specified
            by self.out_file. If "return", the output HTML will be
            directly returned by this function.
        """
        if len(self.similarity_matrix) == 0:
            logging.error("Cannot generate report: no files compared")
            return

        code_list = self.get_copied_code_list()

        # --- 변경(대칭 행렬 생성) ---
        s_ab = self.similarity_matrix[:,:,0]
        s_ba = self.similarity_matrix[:,:,1]
        plot_mtx = np.full_like(s_ab, np.nan, dtype=np.float64)

        # 유효한 페어만 대칭 점수로 채우기
        valid = (s_ab >= 0) & (s_ba >= 0)
        # 벡터화된 대칭 점수 계산
        with np.errstate(divide='ignore', invalid='ignore'):
            # dice 기본
            plot_mtx[valid] = (2 * s_ab[valid] * s_ba[valid]) / (s_ab[valid] + s_ba[valid])

        # 이후 코드(컬러바/히스토그램)는 그대로 plot_mtx 사용
        plt.imshow(plot_mtx)
        plt.colorbar()
        plt.tight_layout()
        sim_mtx_buffer = io.BytesIO()
        plt.savefig(sim_mtx_buffer)
        sim_mtx_buffer.seek(0)
        sim_mtx_base64 = base64.b64encode(sim_mtx_buffer.read()).decode()
        plt.close()

        scores = plot_mtx[~np.isnan(plot_mtx)]
        plt.hist(scores, bins=20)
        plt.tight_layout()
        sim_hist_buffer = io.BytesIO()
        plt.savefig(sim_hist_buffer)
        sim_hist_buffer.seek(0)
        sim_hist_base64 = base64.b64encode(sim_hist_buffer.read()).decode()
        plt.close()

        # render template with jinja and save as html
        with importlib.resources.open_text(
            data_files, "report.html", encoding="utf-8"
        ) as template_fp:
            template = Template(template_fp.read())

        flagged = (plot_mtx > self.conf.display_t)
        flagged_file_count = np.sum(np.any(flagged, axis=1))

        formatted_conf = json.dumps(self.conf.to_json(), indent=4)
        output = template.render(config_params=formatted_conf,
                                 css_files=self.conf.css_files,
                                 version=__version__,
                                 test_count=len(self.test_files),
                                 test_files=self.test_files,
                                 compare_count=len(self.ref_files),
                                 compare_files=self.ref_files,
                                 flagged_file_count=flagged_file_count,
                                 code_list=code_list,
                                 sim_mtx_base64=sim_mtx_base64,
                                 sim_hist_base64=sim_hist_base64)

        if output_mode == "save":
            html_out = output
            # (선택) 주석 전용 표 섹션만 유지하고 싶으면 아래 한 줄만 남기세요
            if self.conf.html_separate_comment_score:
                html_out = self._inject_comment_section(html_out)

            with open(self.conf.out_file, "w", encoding="utf-8") as report_f:
                report_f.write(html_out)

            if not self.conf.silent:
                print(f"Output saved to {self.conf.out_file.replace('//', '/')}")

            if self.conf.autoopen:
                webbrowser.open('file://' + str(Path(self.conf.out_file).resolve()))

        elif output_mode == "return":
            return output
        else:
            raise ValueError("output_mode not supported")

    def _inject_comment_section(self, html_str: str) -> str:
        """
        HTML 리포트 하단에 'Comment-only Similarity' 섹션 표를 추가한다.
        (self.conf.html_separate_comment_score 가 True일 때만 호출됨)
        """
        rows = []
        for (t, r), _ in self.slice_matrix.items():
            f1 = self.file_data.get(t); f2 = self.file_data.get(r)
            if not f1 or not f2:
                continue

            mode = getattr(self.conf, "comment_mode", "kgram")
            if mode == "cosine":
                c1, c2 = getattr(f1, "comment_text", ""), getattr(f2, "comment_text", "")
                if not c1 or not c2:
                    continue
                n = getattr(self.conf, "comment_ngram_n", 3)
                s = comment_cosine_sim(c1, c2, n=n)
                # 코사인은 방향성 없으니 양쪽에 동일 점수 표기
                rows.append((t, r, float(s), float(s), int(s * min(len(c1), len(c2)))))
                continue  # 코사인 처리 끝났으면 k-gram 분기는 건너뜀
            
            if not hasattr(f1, "comment_hashes") or not hasattr(f2, "comment_hashes"):
                continue
            if len(getattr(f1, "comment_hashes", set())) == 0 or len(getattr(f2, "comment_hashes", set())) == 0:
                continue

            oc1, oc2 = find_fingerprint_overlap(
                f1.comment_hashes, f2.comment_hashes, f1.comment_hash_idx, f2.comment_hash_idx
            )
            s1 = get_copied_slices(oc1, f1.k_comment)
            s2 = get_copied_slices(oc2, f2.k_comment)

            ovl1 = int(np.sum(s1[1]-s1[0])) if s1.shape[0] > 0 else 0
            ovl2 = int(np.sum(s2[1]-s2[0])) if s2.shape[0] > 0 else 0
            den1 = max(1, len(getattr(f1, "comment_text", "")))
            den2 = max(1, len(getattr(f2, "comment_text", "")))
            sim1 = ovl1 / den1 if den1 else 0.0
            sim2 = ovl2 / den2 if den2 else 0.0

            rows.append((t, r, sim1, sim2, max(ovl1, ovl2)))

        if not rows:
            return html_str

        table_rows = "\n".join(
            f"<tr><td>{t}</td><td>{r}</td>"
            f"<td>{sim1:.4f}</td><td>{sim2:.4f}</td><td>{ovl}</td></tr>"
            for (t, r, sim1, sim2, ovl) in rows
        )
        section = f"""
        <section style="margin:24px 16px">
            <h2>Comment-only Similarity (분리 스코어)</h2>
            <p>아래 점수는 <b>주석 텍스트만</b>으로 계산한 유사도입니다.</p>
            <table border="1" cellspacing="0" cellpadding="6">
            <thead><tr>
                <th>Test File</th><th>Ref File</th>
                <th>Test Comment Sim</th><th>Ref Comment Sim</th>
                <th>Overlap tokens</th>
            </tr></thead>
            <tbody>
                {table_rows}
            </tbody>
            </table>
        </section>
        """
        insert_at = html_str.rfind("</body>")
        if insert_at == -1:
            return html_str + section
        return html_str[:insert_at] + section + html_str[insert_at:]

    from html import escape

    def render_ipynb_like_html(cells):
        """
        cells: [{"type": "code"|"markdown", "source": str}, ...]
        리턴: 노트북처럼 보이는 HTML 문자열 (각 셀은 <div class="cell ...">)
        """
        out = []
        out.append('<div class="notebook">')
        for i, c in enumerate(cells):
            ctype = c["type"]
            out.append(f'<div class="cell {ctype}">')
            out.append(f'<div class="cell-header">[cell {i}] {ctype}</div>')
            if ctype == "code":
                out.append('<div class="input"><pre>')
                out.append(escape(c["source"]))
                out.append('</pre></div>')
            else:
                out.append('<div class="markdown">')
                # 아주 단순 렌더: 줄바꿈만 보존
                out.append("<pre>" + escape(c["source"]) + "</pre>")
                out.append('</div>')
            out.append('</div>')
        out.append('</div>')
        return "\n".join(out)

    def ipynb_global_to_cell_ranges(offset_map, gstart, gend):
        """
        전역 [gstart, gend) → [(cell_idx, local_line_start, local_line_end)] 목록으로 변환
        여기서는 단순화: 매칭 구간이 하나의 셀 범위를 넘지 않는다고 가정.
        (필요하면 다중 셀跨 매칭을 분할해 반환하도록 확장 가능)
        """
        results = []
        for s, e, ci, lstart, lend in offset_map:
            if gstart >= s and gend <= e:
                results.append((ci, lstart, lend))  # 최소 버전: 라인 범위를 대략 셀 전체로
                break
        return results

    # detector.py (CopyDetector 클래스 내부)
    from pathlib import Path
    import os

    def _to_rel_url(self, abs_path: str) -> str:
        """
        report.html 이 위치한 폴더 기준의 상대경로를 반환 (iframe src로 쓰기 좋게 / 로 정규화)
        """
        report_dir = Path(self.conf.out_file).parent.resolve()
        p_abs = Path(abs_path).resolve()
        try:
            rel = os.path.relpath(p_abs, report_dir)
        except ValueError:
            # (윈도우에서 드라이브가 다르면) 파일명만 사용
            rel = p_abs.name
        return rel.replace(os.sep, "/")

    def _merge_spans(self, spans, join_if_gap=1):
        """겹치거나 거의 붙은 스팬 병합. score는 최대값 유지."""
        if not spans:
            return []
        spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
        merged = [dict(spans[0])]
        for s in spans[1:]:
            last = merged[-1]
            if s["start"] <= last["end"] + join_if_gap:
                last["end"]   = max(last["end"], s["end"])
                last["score"] = max(last.get("score", 0.0), s.get("score", 0.0))
            else:
                merged.append(dict(s))
        return merged

    def _render_highlighted_html(self, src_text, spans, clamp=True):
        """
        src_text: 원문 소스코드(문서 전체 문자열)
        spans: [{"start":int,"end":int,"score":float}, ...]   # 절대 오프셋
        반환: <pre><code>...</code></pre> HTML
        """
        import html
        if not spans:
            return f"<pre class='code'><code>{html.escape(src_text)}</code></pre>"

        n = len(src_text)
        norm = []
        for s in spans:
            st = max(0, min(n, s["start"])) if clamp else s["start"]
            ed = max(0, min(n, s["end"]))   if clamp else s["end"]
            if ed > st:
                sc = float(s.get("score", 1.0))
                sc = 0.0 if sc < 0.0 else (1.0 if sc > 1.0 else sc)
                norm.append({"start": st, "end": ed, "score": sc})

        norm = self._merge_spans(norm, join_if_gap=0)
        # 오른쪽 → 왼쪽(내림차순)으로 삽입: 인덱스 밀림 방지
        norm.sort(key=lambda s: (s["start"], s["end"]), reverse=True)

        out = src_text
        for s in norm:
            st, ed, sc = s["start"], s["end"], s["score"]
            frag = html.escape(out[st:ed])
            mark = f"<span class='hl' data-score='{sc:.3f}'>{frag}</span>"
            out = out[:st] + mark + out[ed:]
        return f"<pre class='code'><code>{out}</code></pre>"


