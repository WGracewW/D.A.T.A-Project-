# The new utilities file to support version 9.0 onwards
# Last edited: July 16, 2026
# Functions surrounded by #** indicate functions that are either new or modified and haven't been tested yet.

def clean_pymupdf_text(text: str) -> str:
    """
    Dynamically clean raw PyMuPDF text from scanned or digital PDFs.

    Detection-first approach: each cleaning stage analyses the text to
    decide whether the pattern is actually present before acting on it.
    Clean/well-typed documents pass through largely untouched.
    """
    import re
    from collections import Counter

    # ------------------------------------------------------------------ #
    # Stage 1 — Page numbers
    # ------------------------------------------------------------------ #

    #_PAGE_NUMBER_RE = re.compile(
    #    r'''^\s*
    #    (
    #        -\s*\d{1,4}\s*-        |   # -6-
    #        \(\s*\d{1,4}\s*\)      |   # (6)
    #        \[\s*\d{1,4}\s*\]      |   # [6]
    #        Page\s+\d{1,4}         |   # Page 6
    #        \d{1,4}\s*/\s*\d{1,4}     # 6/14  (current/total)
    #    )
    #    \s*$''',
    #    re.IGNORECASE | re.VERBOSE,
    #)

    _PAGE_OF_RE = re.compile(
    r'\bpage\s+\d{1,4}\s+of\s+\d{1,4}\b',
    re.IGNORECASE
    )

    def _isolate_page_of(text: str) -> str:
        # Surround matches with newlines so they become standalone lines
        return _PAGE_OF_RE.sub(r'\n\g<0>\n', text)
    #def _remove_page_numbers(lines: list[str]) -> list[str]:
    #    matches = [i for i, ln in enumerate(lines) if _PAGE_NUMBER_RE.match(ln)]
    #    if len(matches) < 2:
    #        return lines
    #    remove = set(matches)
    #    return [ln for i, ln in enumerate(lines) if i not in remove]


    # ------------------------------------------------------------------ #
    # Stage 2 — Repeated header / footer blocks
    # ------------------------------------------------------------------ #

    def _remove_repeated_header_footer_blocks(
        lines: list[str],
        min_repeats: int = 4,
        block_radius: int = 3,
    ) -> list[str]:
        if len(text.split()) > 400: # ONLY use on MULTI-PAGE TEXTS!
            non_empty_indices = [i for i, ln in enumerate(lines) if ln.strip()]

            def normalise(s: str) -> str:
                return re.sub(r'\s+', ' ', s.strip().lower())

            ngram_hits: Counter = Counter()
            ngram_line_sets: dict[str, list[list[int]]] = {}

            for size in range(1, block_radius + 1):
                for start in range(len(non_empty_indices) - size + 1):
                    idx_group = non_empty_indices[start: start + size]
                    key = '\x00'.join(normalise(lines[i]) for i in idx_group)
                    if not key.strip('\x00'):
                        continue
                    ngram_hits[key] += 1
                    ngram_line_sets.setdefault(key, []).append(idx_group)

            remove: set[int] = set()
            for key, count in ngram_hits.items():
                if count < min_repeats:
                    continue
                parts = key.split('\x00')
                if any(len(p) > 120 for p in parts):
                    continue
                for idx_group in ngram_line_sets[key]:
                    remove.update(idx_group)

            return [ln for i, ln in enumerate(lines) if i not in remove]
        else: # do nothing
            return lines

    # ------------------------------------------------------------------ #
    # Stage 3 — Table-of-contents lines
    # ------------------------------------------------------------------ #

    # FIX: Replaced nested/ambiguous quantifiers with a simple anchored pattern.
    # Old: r'(?:(?:\.{2,}|\s*\.\s*){2,})\s*\d{1,3}\s*$'  — catastrophic backtracking
    # New: require 3+ consecutive dots (real TOC leaders always are), no nesting.
    _TOC_RE = re.compile(r'\.{3,}\s*\d{1,3}\s*$')

    # FIX: Replaced the combined lazy+greedy r'.{5,}?\s{3,}\d' pattern with a
    # two-step function. The old single-regex form backtracked catastrophically
    # on long lines that partially matched but had no 3-space run before the number.
    def _is_toc_numbered(line: str) -> bool:
        """'2.2   Some Title   8' style TOC line, checked in two safe passes."""
        if len(line) > 300:
            return False
        if not re.match(r'^\s*\d+(?:\.\d+)*\s+', line):
            return False
        return bool(re.search(r'\s{3,}\d{1,3}\s*$', line))

    def _remove_toc_lines(lines: list[str]) -> list[str]:
        toc_indices = [
            i for i, ln in enumerate(lines)
            if _TOC_RE.search(ln) or _is_toc_numbered(ln)
        ]
        if len(toc_indices) < 3:
            return lines

        clustered = _has_cluster(toc_indices, window=30, min_in_window=3)
        if not clustered:
            return lines

        remove = set(toc_indices)
        return [ln for i, ln in enumerate(lines) if i not in remove]


    # ------------------------------------------------------------------ #
    # Stage 4 — Junk / noise tokens
    # ------------------------------------------------------------------ #

    _JUNK_CHECKS = [
        re.compile(r'[\x00-\x08\x0b-\x1f\x7f]'),
        re.compile(r'\^[A-Z@\[\\\]^_]'),
    ]

    def _is_junk_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        for pattern in _JUNK_CHECKS:
            if pattern.search(stripped):
                return True
        non_space = stripped.replace(' ', '')
        if len(non_space) == 0:
            return False
        symbol_count = sum(1 for c in non_space if not c.isalnum())
        if len(non_space) <= 6 and symbol_count / len(non_space) >= 0.5:
            return True
        return False

    def _remove_junk_tokens(lines: list[str]) -> list[str]:
        flagged = [i for i, ln in enumerate(lines) if _is_junk_line(ln)]
        if not flagged:
            return lines
        remove = set(flagged)
        return [ln for i, ln in enumerate(lines) if i not in remove]


    # ------------------------------------------------------------------ #
    # Stage 5 — Hyphenated line-breaks
    # ------------------------------------------------------------------ #

    _SOFT_HYPHEN_RE = re.compile(r'[¬\xad]-?\n\s*')
    _HARD_HYPHEN_RE = re.compile(r'(?<=[a-z])-\n([a-z])')

    def _fix_hyphenated_linebreaks(text: str) -> str:
        if '\xad' in text or '¬' in text:
            text = _SOFT_HYPHEN_RE.sub('-', text)
        if re.search(r'[a-z]-\n[a-z]', text):
            text = _HARD_HYPHEN_RE.sub(r'\1', text)
        return text


    # ------------------------------------------------------------------ #
    # Stage 6 — Over-spaced words
    # ------------------------------------------------------------------ #

    _SPACED_WORD_RE = re.compile(r'\b(?:[a-z] ){2,}[a-z]\b')

    def _fix_overspacedwords(text: str) -> str:
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return text
        single_char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
        if single_char_ratio < 0.15:
            return text

        def _collapse(m: re.Match) -> str:
            return m.group(0).replace(' ', '')

        return _SPACED_WORD_RE.sub(_collapse, text)


    # ------------------------------------------------------------------ #
    # Stage 7 — Soft line-wrap rejoining
    # ------------------------------------------------------------------ #

    def _rejoin_wrapped_lines(text: str) -> str:
        text = re.sub(r'\n{2,}', '\x00PARA\x00', text)
        lines = text.split('\n')

        wrapped_count = 0
        for i, ln in enumerate(lines[:-1]):
            ln_s = ln.rstrip()
            next_s = lines[i + 1].lstrip()
            if (ln_s
                    and next_s
                    and not re.search(r'[.!?:;]\s*$', ln_s)
                    and re.match(r'[a-z("]', next_s)):
                wrapped_count += 1

        wrap_ratio = wrapped_count / max(len(lines), 1)

        if wrap_ratio >= 0.25:
            text = re.sub(r'(?<![.!?:;])\n(?=[a-z("])', ' ', text)

        text = text.replace('\x00PARA\x00', '\n\n')
        return text


    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def _has_cluster(
        indices: list[int],
        window: int,
        min_in_window: int,
    ) -> bool:
        if not indices:
            return False
        for i, start in enumerate(indices):
            count = sum(1 for j in indices[i:] if j - start <= window)
            if count >= min_in_window:
                return True
        return False


    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _isolate_page_of(text)
    lines = text.split('\n')

    #lines = _remove_page_numbers(lines)
    lines = _remove_repeated_header_footer_blocks(lines)
    lines = _remove_toc_lines(lines)
    lines = _remove_junk_tokens(lines)

    text = '\n'.join(lines)

    text = _fix_hyphenated_linebreaks(text)
    text = _fix_overspacedwords(text)
    text = _rejoin_wrapped_lines(text)

    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def clean_prompt_input(prompt: str) -> str:
    """
    Post-interpolation cleanup for LLM prompts containing PyMuPDF-extracted
    text injected into a structured template.

    Fixes the common typo EXERPT -> EXCERPT, then isolates the text between
    the BEGIN/END EXCERPT delimiters and cleans only that portion. Everything
    outside the delimiters (system prompt, instructions, few-shots) is
    left completely untouched.

    Returns the prompt with the excerpt portion cleaned.
    """
    import re 

    def _clean_excerpt(text: str) -> str:
        """
        Master cleaning pipeline for the extracted study text.

        Applies each stage in order:
        1. TOC block removal - removed from code.
        2. Structural noise line removal
        3. Orphan fragment removal
        4. Blank line normalisation

        Each stage is gated — it will do nothing if the relevant pattern
        is not sufficiently present in the text (safe for clean documents).
        """

        # Stage 2: Remove isolated structural metadata lines (section numbers,
        # all-caps labels, lone page numbers, stray punctuation)
        lines = text.split('\n')
        lines = _remove_structural_noise_lines(lines)
        text  = '\n'.join(lines)

        # Stage 3: Remove severely truncated / orphaned sentence fragments
        text = _remove_orphan_fragments(text)

        # Stage 4: Collapse runs of 3+ blank lines down to a single blank line
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    # ------------------------------------------------------------------ #
    # Stage 2 — Structural noise line removal
    # ------------------------------------------------------------------ #

    # Each pattern targets a specific category of noise line.
    # Kept as separate regexes (rather than one combined verbose regex)
    # to avoid quote/escape conflicts inside character classes.

    # Bare section numbers with no content: "3", "3.1", "2.2.1"
    _BARE_SECTION_NUM_RE = re.compile(r'^\d+(?:\.\d+)*\s*$')

    # Lines that are entirely capital letters and spaces: "ARCHIVES", "TEST SYSTEM"
    # Minimum 3 chars to avoid matching single-letter noise
    _ALL_CAPS_LABEL_RE   = re.compile(r'^[A-Z][A-Z\s]{2,}\s*$')

    # Lines that contain only a 1-3 digit number (lone page references)
    _LONE_PAGE_NUM_RE    = re.compile(r'^\d{1,3}\s*$')

    # Lines that are only dashes, quotes, or similar stray punctuation characters
    # Unicode escapes used to avoid embedding literal quote chars in the string:
    #   \u2013 = en dash, \u2014 = em dash
    #   \u2018/\u2019 = curly single quotes, \u201c/\u201d = curly double quotes
    _STRAY_PUNCT_RE      = re.compile(
        r'^[\-\u2013\u2014\'\"\u2018\u2019\u201c\u201d]{1,3}\s*$'
    )

    # Safeguard: these toxicology-relevant ALL-CAPS terms should never be removed
    # even if they match one of the patterns above (e.g. "DERMAL", "ORAL")
    _KEEP_LABELS_RE = re.compile(
        r'\b(DERMAL|ORAL|INHALATION|TOPICAL|INTRADERMAL|OCCLUSIVE|'
        r'INDUCTION|CHALLENGE|DOSE|RESULT|CONCLUSION|SUMMARY)\b'
    )


    def _is_structural_noise(line: str) -> bool:
        """
        Return True if this line is isolated structural metadata that adds
        no semantic value to the prompt (e.g. a bare section number, a
        lone page number, an all-caps document label).

        Always returns False for lines containing toxicology-relevant terms
        so that meaningful content is never accidentally stripped.
        """
        s = line.strip()
        if not s:
            return False  # blank lines handled separately by blank-line normalisation

        # Never remove lines that contain domain-relevant keywords
        if _KEEP_LABELS_RE.search(s):
            return False

        return bool(
            _BARE_SECTION_NUM_RE.match(s)
            or _ALL_CAPS_LABEL_RE.match(s)
            or _LONE_PAGE_NUM_RE.match(s)
            or _STRAY_PUNCT_RE.match(s)
        )


    def _remove_structural_noise_lines(lines: list[str]) -> list[str]:
        """
        Filter out every line identified as structural noise.
        Operates on the full line list so context (surrounding lines) is
        available to the individual check if needed in future.
        """
        return [ln for ln in lines if not _is_structural_noise(ln)]


    # ------------------------------------------------------------------ #
    # Stage 3 — Orphan fragment removal
    # ------------------------------------------------------------------ #

    # These patterns identify lines that are almost certainly NOT fragments:
    _TERMINAL_PUNCT_RE = re.compile(r'[.!?:;,]$')          # ends with punctuation
    _NUMBERED_ITEM_RE  = re.compile(r'^\d+[.)]')            # numbered list item
    _FUNCTION_WORDS_RE = re.compile(                         # contains common English words
        r'\b(the|a|an|of|in|and|or|to|was|were|is|are)\b',
        re.IGNORECASE
    )
    _DOSE_RE = re.compile(                                   # contains a dose/concentration
        r'\d+%|\d+\s*mg|\d+\s*ml',
        re.IGNORECASE
    )


    def _is_orphan_fragment(line: str) -> bool:
        """
        Return True if this line looks like a severely truncated or
        incomplete sentence fragment left over from OCR misreads or
        mid-page cuts — i.e. content that will confuse rather than
        help the model.

        A line is considered an orphan fragment if it is:
        - Short (under 10 chars)
        - Does not end with terminal punctuation
        - Is not a numbered list item
        - Contains no common English function words
        - Contains no dose/concentration patterns
        """
        s = line.strip()

        # Empty or long lines are never fragments
        if not s or len(s) > 10:
            return False

        # Lines ending with punctuation are complete thoughts
        if _TERMINAL_PUNCT_RE.search(s):
            return False

        # Numbered list items are structural but intentional
        if _NUMBERED_ITEM_RE.match(s):
            return False

        # If the line contains common function words it's likely real prose
        if _FUNCTION_WORDS_RE.search(s):
            return False

        # Dose/concentration values are always meaningful — keep them
        if _DOSE_RE.search(s):
            return False

        return True


    def _remove_orphan_fragments(text: str) -> str:
        """
        Remove orphan fragment lines, but only when the document contains
        enough of them to suggest it is a noisy scan (gate: >= 3 fragments).

        For clean documents this function is a no-op.
        """
        lines   = text.split('\n')
        flagged = [i for i, ln in enumerate(lines) if _is_orphan_fragment(ln)]

        # Gate: don't remove anything unless there are at least 3 fragments
        # (avoids false positives in clean documents)
        if len(flagged) < 3:
            return text

        remove = set(flagged)
        return '\n'.join(ln for i, ln in enumerate(lines) if i not in remove)


    # ------------------------------------------------------------------ #
    # Shared utility
    # ------------------------------------------------------------------ #

    def _has_cluster(indices: list[int], window: int, min_in_window: int) -> bool:
        """
        Return True if at least `min_in_window` values in `indices` fall
        within any contiguous span of `window` lines.

        Used to confirm that matched lines are grouped together (e.g. a real
        TOC block) rather than scattered randomly across the document.
        """
        for i, start in enumerate(indices):
            count = sum(1 for j in indices[i:] if j - start <= window)
            if count >= min_in_window:
                return True
        return False
    
    # Cleaning

    # Fix the delimiter typo that appears in the original prompt template
    prompt = prompt.replace('BEGIN EXERPT', 'BEGIN EXCERPT')
    prompt = prompt.replace('END EXERPT', 'END EXCERPT')

    # Match everything between the dashed delimiter lines, capturing:
    #   group(1) = opening delimiter  e.g. "BEGIN EXCERPT-------"
    #   group(2) = the raw extracted text we want to clean
    #   group(3) = closing delimiter  e.g. "-------END EXCERPT"
    delimiter_re = re.compile(
        r'(BEGIN EXCERPT-+)(.*?)(-+END EXCERPT)',
        re.DOTALL
    )
    match = delimiter_re.search(prompt)
    if not match:
        # No excerpt block found — return the prompt unchanged
        return prompt

    before  = prompt[:match.start()]   # everything before the opening delimiter
    open_d  = match.group(1)           # the opening delimiter line itself
    excerpt = match.group(2)           # the injected study text
    close_d = match.group(3)           # the closing delimiter line itself
    after   = prompt[match.end():]     # everything after the closing delimiter

    excerpt = _clean_excerpt(excerpt)

    # Reassemble the prompt with the cleaned excerpt in place
    return before + open_d + excerpt + close_d + after

def cleanup_text(text: str) -> str:
    """
    Removes footers, page numbers, tables, and irrelevant statistical
    information from extracted PDF text.

    Args:
        text: Raw extracted text string.

    Returns:
        Cleaned text string.
    """
    import re

    header_patterns = [
        r"^\s*(Appendix No\. \d+):.*",
        r"^\s*Table \d+.*",
        r"^\s*Figure \d+.*",
    ]
    footer_patterns = [
        r"Page \d+ of \d+",
        r"^\s*\d+\s*$",
    ]
    table_patterns = [
        r"^\s*(\d+\.\d+\s+){4,}",
        r"^\s*(\d+\s+){4,}",
    ]
    stat_patterns = [
        r"p\s*[<>]?\s*0\.\d+",
        r"\d+\s*±\s*\d+",
        r"[A-Za-z]+\s*=\s*\d+",
        r"r\s*=\s*-?\d+\.\d+",
    ]
    table_keywords = {
        "SD", "SEM", "N.S.", "p<", "p>", "p=", "t-test", "ANOVA",
        "±", "SE", "STD", "STDEV", "standard deviation", "standard error",
        "statistical", "TABLE",
    }

    lines = text.split("\n")
    cleaned_lines = []
    in_table = False
    table_lines = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if any(re.match(p, line) for p in header_patterns):
            continue
        if any(re.match(p, line) for p in footer_patterns):
            continue
        if line.isupper() and len(line) > 3:
            continue

        is_table_line = (
            any(re.match(p, line) for p in table_patterns)
            or any(kw.lower() in line.lower() for kw in table_keywords)
            or any(re.search(p, line) for p in stat_patterns)
            or line.count("\t") > 2
            or line.count("  ") > 3
        )

        if is_table_line:
            in_table = True
            table_lines += 1
            continue

        if in_table and (len(line.split()) > 5 or len(line) > 60):
            in_table = False
            table_lines = 0

        if not in_table:
            line = re.sub(r"\([^)]*\b(p|SD|SE|SEM|n)\b[^)]*\)", "", line)
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\b(SD|SE|SEM)\b\s*[<>]?=\s*\d+\.?\d*", "", result)
    result = re.sub(r"\d+\s*±\s*\d+", "", result)
    result = re.sub(r"\s+", " ", result).strip()

    return result

def is_table(text:str,min_numbers:int = 3, min_multilines:int = 3, min_decimal_count:int = 8) -> bool:
    """
    Detects tables.
    Min numbers = the minimum amount of numbers present in a line to be considered 'numeric heavy'.
    Min multilines = the minimum number of short lines separated by \n to be considered 'table-like'
    min_ecimal_count = the minimum amount of decimals present in the text to classify as table.
    """
    import re

    non_blank_lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not non_blank_lines:
        return False

    # -------------------------------------------------------------
    # Detect Tables
    # -------------------------------------------------------------
    table_pattern = re.compile(r'\btables?\s+\d+(?:\s*-\s*\d+)?\b', re.IGNORECASE) # 'table _' or 'tables _ - _' or 'tables _-_' or 'table _-_' ...

    if len(table_pattern.findall(text))>=1: # text contains a table title
        return True

    # Count decimal values
    decimal_count = len(re.findall(r"\b\d+\.\d+\b", text))

    # Lines containing many numbers
    numeric_heavy = sum(
        len(re.findall(r"\d", line)) >= min_numbers
        for line in non_blank_lines
    )

    multi_number_lines = sum(
        len(re.findall(r"\d+", line)) >= min_multilines
        for line in non_blank_lines
    )

    # Statistical notation (P-values)
    has_stats = bool(
        re.search(r"\b[pP]\s*[<>]=?\s*0?\.\d+", text)
    )
    def _has_short_line_run(lines, max_len=30, run_len=5, digit_frac=0.4):
        run = 0
        digit_lines = 0
        for line in lines:
            if len(line) <= max_len:
                run += 1
                if re.search(r"\d", line):
                    digit_lines += 1
                if run >= run_len and digit_lines / run >= digit_frac:
                    return True
            else:
                run = 0
                digit_lines = 0
        return False

    looks_like_table = (
        decimal_count > min_decimal_count
        or has_stats
        or numeric_heavy / len(non_blank_lines) > 0.50
        or multi_number_lines / len(non_blank_lines) > 0.40
        or _has_short_line_run(non_blank_lines)
    )

    if looks_like_table:
        return True
#**
def detect_sections(pdf_fp:str | None, page:int | None = None, text:int | None = None, target_titles:list|None=None,searching:bool=False):
    """
    Inputs:
    pdf_fp: str
    page: optional, specify a single page in the pdf to search, 1-based!!
    text: optional, specify a single string of text (i.e. a page content) to search.
    searching:bool, if true, must provide a list of target titles to search
    target_titles: list of string

    Outputs:
    Sections - List of Section Objects

    if searching == True, the search returns first-hits only
    """
    import re
    import fitz
    import unicodedata

    sections = []
    debug = []

    if page and text:
        raise ValueError("When using detect_sections, please only specify the page number OR the text string you wish to use! Do not input both.")
    if pdf_fp and text:
        raise ValueError("When using detect_sections, please only specify the PDF OR the text string you wish to use! Do not input both. Specifying the text string will only return sections present in the text string.")

    class Section:
        def __init__(self, page_num:int, content:str, title:str):
            self.page_num = page_num
            self.content = content
            self.title = title
    
    Title_patterns = r"^\s*(?:[A-Z][A-Z0-9&''\-]*\s+){0,3}[A-Z][A-Z0-9&''\-]*\s*(?:\((?:cont(?:\.|d|inued)?|continued|contd?)\))?\s*$" # All cap words followed by opt. (cont.) or the like.
    
    TOC_TITLE_RE = re.compile(
        r'(?:(?<=\n)|(?<=\A))'
        r'[ \t]*'
        r'('
            r'(?=.*[A-Za-z])'
            r'(?:[A-Z][a-zA-Z]*|[a-z]{1,4})'
            r'(?:[ \t]+(?:[A-Z][a-zA-Z]*|[a-z]{1,4}))*'
            r'|'
            r'(?=.*[A-Za-z])'
            r'[A-Z0-9][A-Z0-9 \t\-–—]{2,}'
        r')'
        r'[ \t]*'
        r'\n+'
        r'\s*'
        r'(?=[A-Z])',
        re.MULTILINE
    )

    with fitz.open(pdf_fp) as doc:
        pages = [page.get_text() for page in doc]
        if page or text:
            if page:
                clean_text = clean_pymupdf_text(pages[int(page-1)])
                page_num = page
            if text:
                page_num = 0
                clean_text = text
            
            # First match the first pattern
            matches1 = re.findall(Title_patterns, clean_text, re.MULTILINE)

            if len(matches1)>0:
                for match in matches1:
                    title = match.strip()
                    
                    if (title):
                        sections.append(
                            Section(
                                page_num=page_num,
                                content=clean_text,
                                title=title
                            )
                        )

            # Then match the second pattern
            for m in TOC_TITLE_RE.finditer(clean_text):
                title=m.group(1).strip()

                if (title):
                        sections.append(
                            Section(
                                page_num=page_num,
                                content=clean_text,
                                title=title
                            )
                        )
            
            return sections

        for page_num, text in enumerate(pages, start=1):
            clean_text = clean_pymupdf_text(text) # First clean the format

            # First match the first pattern
            matches1 = re.findall(Title_patterns, clean_text, re.MULTILINE)

            if len(matches1)>0:
                for match in matches1:
                    title = match.strip()
                    
                    if (title):
                        sections.append(
                            Section(
                                page_num=page_num,
                                content=clean_text,
                                title=title
                            )
                        )
            
            # Then match the second pattern
            for m in TOC_TITLE_RE.finditer(clean_text):
                title=m.group(1).strip()

                if (title):
                        sections.append(
                            Section(
                                page_num=page_num,
                                content=clean_text,
                                title=title
                            )
                        )

        if searching == False: # Simply return a list of all section objects detected in the pdf
            return sections
        
        elif searching == True:
            if target_titles:
                # Search for the relevant sections and return them
                target_sections = []
                for target in target_titles:
                    for section in sections:
                        if (target.lower() in section.title.lower()) and (section.content not in [t.content for t in target_sections]):
                            if is_toc(section.content):
                                continue # skip TOC pages
                            target_sections.append(section)

                if len(target_sections)==0:
                    debug.append(f"NO target sections found")
                    return target_sections
                    
                debug.append([t.title for t in target_sections])
                organized_target_sections = []
                for section in target_sections:
                    if section.title.lower() in [s.title.lower() for s in organized_target_sections]: # matching section titles between analyzed section and pre-covered sections
                        continue # move on to next one; already covered.
                    if any(s.title.lower() in section.title.lower() for s in organized_target_sections): # any pre-covered section headers as substring in the section title that is being analyzed.
                        continue # move on to next one; already covered.

                    page_num = section.page_num

                    # First and foremost detect if the page is a TOC
                    if is_toc(section.content):
                        continue # do not add to organized_target_sections if page is a TOC
                    
                    # Second detect if the page fully contains the section (includes multiple sections)
                    section_hot_words = ["introduction","1ntroduction","summary","sumnary","abstract","test substance","test material","procedure","results","discussion","references","methods","cell line","harvest","staining","coding"]
                    excluded_hot_words = [word for word in section_hot_words if word not in set(target_titles)]

                    other_matches = re.findall(Title_patterns, pages[page_num-1], re.MULTILINE)
                    other_matches2 = [m.group(0).strip() for m in TOC_TITLE_RE.finditer(pages[page_num-1])]
                    combined_matches = set(other_matches + other_matches2)

                    if (len(combined_matches)) > 1: # More than one section object exists for this page
                        content = section.content
                        target_title = None
                        non_target_titles = []
                        for title in combined_matches:
                            if ( re.compile(r'\b(?=\S*[A-Za-z])(?=\S*\d)(?=\S*[^A-Za-z0-9])\S+\b').search(title) ):
                                # if title is junk (both alphanumeric characters and multiple special characters.)
                                continue
                            if (title.lower() in excluded_hot_words) or any(word.lower() in title.lower() for word in excluded_hot_words):
                                # if title is in elcuded hot words, or any excluded hot words are in title
                                non_target_titles.append(title)
                                continue
                            if (title.lower() in target_titles) or any(word.lower() in title.lower() for word in target_titles):
                                # If this title is a target title
                                target_title = title

                        if (len(non_target_titles)>0) and target_title: # Non target section titles exist on this page
                            debug.append(f"NON target section titles exist here:{non_target_titles}")
                            # Now remove text preceeding our target title and proceeding any non-target titles
                            match = re.search(re.escape(target_title), content, re.IGNORECASE)
                            if match: # Cut everything before the target title
                                debug.append(f"start of target section: {match.start()}")
                                content = content[match.start():]

                            # Cut everything from the first non-target title onward
                            for title in non_target_titles:
                                match2 = re.search(r'(?<!\S)' + re.escape(title.strip()) + r'(?!\S)', content, re.IGNORECASE)
                                if match2:
                                    if ( match2.start() > match.start() ) and ( match2.start()>=150): # ONLY if at least 150 letters have elapsed.
                                        debug.append(f"start of non-target section: {match2.start()}")
                                        try:
                                            content = content[:match2.start()]
                                    
                                        except:
                                            pass

                            organized_target_sections.append(
                                Section(
                                    page_num=page_num,
                                    content=content,
                                    title=target_title
                                )
                            )
                            continue # if non-target titles exist, then the summary section must have ended on this page

                        else: # Only the target section title exists on this page
                            organized_target_sections.append(section)

                    else: # Only one section header exists here, including non-sensical ones.
                        organized_target_sections.append(section)

                    # Checking subsequent page ...
                    if page_num >= len(pages): # make sure following page exists
                        continue
                    if page_num+1 in [t.page_num for t in organized_target_sections]: # subsequent page already included in the retrieved pages (i.e. it contains a target title)
                        continue

                    next_page = pages[page_num] # Since pages is 0-based, page_num should be 1 page after the target page
                    clean_next_page = clean_pymupdf_text(next_page)

                    # Find titles in next page
                    next_matches = re.findall(Title_patterns, clean_next_page, re.MULTILINE)
                    next_matches2 = [m.group(1).strip() for m in TOC_TITLE_RE.finditer(clean_next_page)]
                    combined_next_matches = next_matches + next_matches2
                    non_target_titles2 = []

                    for line in clean_next_page.split('\n'): # search line by line
                        line = line.strip()
                        if ( len(line.split()) > 10 ): # first sentence
                            debug.append(line)
                            if line[0].islower(): # first sentence is lowercase
                                debug.append(f"IS LOWER: {page_num+1}")
                                if (len(next_matches)+len(next_matches))>0: # More than 1 title exists on the next page
                                    content2 = None

                                    for title in combined_next_matches:
                                        if ( re.compile(r'\b(?=\S*[A-Za-z])(?=\S*\d)(?=\S*[^A-Za-z0-9])\S+\b').search(title) ):
                                            # if title is junk (both alphanumeric characters and multiple special characters.)
                                            continue
                                        if (title.strip().lower() in excluded_hot_words) or any(word.strip().lower() in title.lower() for word in excluded_hot_words):
                                            # if title is in elcuded hot words, or any excluded hot words are in title
                                            non_target_titles2.append(title)
                                            continue

                                    if (len(non_target_titles2)>0): # Non target section titles exist on this page

                                        # Cut everything from the first non-target title onward
                                        for title in non_target_titles2:
                                            match = re.search(r'(?<!\S)' + re.escape(title.strip()) + r'(?!\S)', clean_next_page, re.IGNORECASE)
                                            if match:
                                                content2 = clean_next_page[:match.start()]

                                        next_page_sec = Section(
                                            page_num=page_num+1,
                                            content=content2,
                                            title=str(f"{section.title}; continued page")
                                        )
                                        if next_page_sec.page_num not in [t.page_num for t in organized_target_sections]:
                                            organized_target_sections.append(next_page_sec)
                                            break

                                else:
                                    next_page_sec = Section(
                                        page_num=page_num+1,
                                        content=clean_next_page,
                                        title=str(f"{section.title}; continued page")
                                    )
                                    organized_target_sections.append(next_page_sec)
                                    break

                            else: # First sentence is not lower case
                                # Find titles in the following page
                                if (len(next_matches)>0) or (len(next_matches2)>0):
                                    if any(t.strip().lower() in excluded_hot_words for t in next_matches) or any(t.strip().lower() in excluded_hot_words for t in next_matches2): # next page contains a new section
                                        break
                                    else:
                                        pass

                                # next page does not begin a new section
                                # If the next page has very little words, assume it is a continuation of the previous page. 
                                word_limit = 50
                                if len(clean_next_page.split()) <= word_limit:
                                    next_page_sec = Section(
                                        page_num = page_num+1,
                                        content = clean_next_page,
                                        title = str(f"{section.title}; continued page")
                                    )
                                    organized_target_sections.append(next_page_sec)
                                    break

                                else:
                                    pass

                                if (page_num+1) not in [o.page_num for o in organized_target_sections]:
                                    PAGE_NUM_RE = re.compile(
                                        r"""
                                        (?:
                                            # Form 1: "Page 2", "Page 2 of 14", case-insensitive
                                            \bPage\s+\d+(?:\s+of\s+\d+)?\b
                                        )
                                        """,
                                        re.VERBOSE | re.IGNORECASE,
                                    )
                                    matches_next_page = PAGE_NUM_RE.findall(clean_next_page)
                                    matches_previous_page = PAGE_NUM_RE.findall(section.content)

                                    if matches_previous_page: # page number exists in the target section
                                        page_num_previous = int(matches_previous_page[0].split()[1]) # both 'page __ of __' and 'page __' has the page number in the second position.
                                        next_page_num = section.page_num+1

                                        while not matches_next_page: # while next page also contains page number; not an insert
                                            debug.append(f"not page numbers found on page {next_page_num}")
                                            next_page_num += 1
                                            if next_page_num > len(pages):
                                                break # no pages left to analyze
                                            matches_next_page = PAGE_NUM_RE.findall(pages[next_page_num-1]) # convert to 0-based

                                        if matches_next_page:
                                            if int(matches_next_page[0].split()[1]) == (page_num_previous+1): # Page numbers are sequential
                                                debug.append(f"Continuation page found on page {next_page_num}")
                                                if next_page_num not in [o.page_num for o in organized_target_sections]:
                                                    next_page_sec = Section(
                                                        page_num = next_page_num,
                                                        content = pages[next_page_num-1],
                                                        title = str(f"{section.title}; continued page")
                                                    )
                                                    organized_target_sections.append(next_page_sec)
                                                    break
                                                else: # page already included in organized target sections
                                                    break
                                    
                                        else: # next page does not contain page number; likely an insert page (i.e. those 'best copy available' inserts)
                                            break

                                    else: # page numbers are not a thing in this document or does not exist in the target section
                                        break # be safe and assume no continuations.

                                else: # Page already included in organized target sections
                                    break

                return organized_target_sections
#**
def is_toc(text: str, min_entries: int = 4, min_ratio: float = 0.2,words_threshold:int = 100) -> bool:
    import re

    non_blank_lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not non_blank_lines:
        return False

    # -------------------------------------------------------------
    # Identify Obvious TOCs
    # -------------------------------------------------------------
    len_words = len(text.split())
    if (len_words < words_threshold) and ('table of contents' in text.lower().strip()):
        return True
    
    # if 'contents' in header
    headers = detect_sections(pdf_fp=None, text=text)
    headers_text = [h.title.lower() for h in headers]
    if any('contents' in h for h in headers_text):
        return True

    # -------------------------------------------------------------
    # Reject obvious tables
    # -------------------------------------------------------------

    table = is_table(text)
    if table:
        return False

    # -------------------------------------------------------------
    # In-depth Positive TOC evidence
    # -------------------------------------------------------------

    # Section numbers like 2.1, 3.4.5
    section_re = re.compile(
        r"""
        ^[\s`'_~•·\-–—]*
        [A-Za-z]{0,2}
        [\s`'_~•·\-–—]*
        \d+
        (?:\s*\.\s*\d+)+
        \s*\.?
        """,
        re.MULTILINE | re.VERBOSE,
    )

    # ALL CAPS headings
    caps_heading_re = re.compile(
        r"^[A-Z][A-Z\s\-/&(),:.]{4,}$",
        re.MULTILINE,
    )

    # TOC entries ending with page numbers
    trailing_page_re = re.compile(
        r"""
        ^
        (?!.*\d.*\d.*\d)          # no more than ~2 numbers on line
        [^\n]{3,80}?
        (?:\.{2,}|\s{2,})
        \d{1,3}
        \s*$
        """,
        re.MULTILINE | re.VERBOSE,
    )

    standalone_page_re = re.compile(
        r"^\s*\d{1,3}\s*$",
        re.MULTILINE,
    )

    numbered = len(section_re.findall(text))
    caps = len(caps_heading_re.findall(text))
    trailing = len(trailing_page_re.findall(text))
    standalone = len(standalone_page_re.findall(text))

    title_like = numbered + caps + trailing

    if title_like >= 4:
        matched = title_like + standalone
    else:
        matched = title_like

    has_contents_header = bool(
        re.search(
            r"^\s*(contents|table of contents)\s*$",
            text,
            re.I | re.M,
        )
    )

    ratio = matched / len(non_blank_lines)

    return has_contents_header or (
        matched >= min_entries
        and ratio >= min_ratio
    )

def ocr_marker(pdf_fp:str, page:int,use_llm:bool=False):
    """
    Marker OCR extraction.
    use_llm=True Uses Ollama's Phi4 model to clean up the text after surya OCR. Optional.
    Enter the page as 1-based!!!
    """
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    if page:
        page_num = f"{page-1}" # Marker takes page numbers as 0-based.

        if use_llm:
            config = {
            "page_range": str(page_num),
            "output_format": "markdown",
            "use_llm": True,
            "llm_service": "marker.services.ollama.OllamaService",
            "ollama_model": "phi4",
            "disable_multiprocessing": True
            }

        else:
            config = {
            "page_range": str(page_num),
            "output_format": "markdown",
            "disable_multiprocessing": True
            }

        config_parser = ConfigParser(config)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer()
        )

        rendered = converter(pdf_fp)

        text, _, images = text_from_rendered(rendered)
    
    else: # no page specified
        raise ValueError("OCR function is called without a page number. To minimize computational time, please specify the page!")

    return text # string, markdown format

def ocr_paddle(
    pdf_path: str,
    pages: int | list[int] | tuple[int, int],
    det_model_dir: str = r"C:\Users\Grace\.paddlex\official_models\PP-OCRv6_medium_det",
    rec_model_dir: str = r"C:\Users\Grace\.paddlex\official_models\PP-OCRv6_medium_rec",
    textline_ori_model_dir: str = r"C:\Users\Grace\.paddlex\official_models\PP-LCNet_x1_0_textline_ori",
    dpi: int = 200,
    lang: str = "en",
    use_textline_orientation: bool = True,
    ) -> dict[int, str]:
    """
    Run PaddleOCR (3.x) on a subset of pages from a PDF, fully offline.

    pages: single page number (0-indexed), list of page numbers,
           or (start, end) tuple treated as an inclusive range.
    *_model_dir: local paths to pre-downloaded PaddleOCR model dirs,
                 required to avoid any network call.

    Returns {page_number: ocr_text}.
    """
    import fitz  # PyMuPDF
    import numpy as np
    from paddleocr import PaddleOCR

    if isinstance(pages, int):
        page_nums = [pages]
    elif isinstance(pages, tuple):
        start, end = pages
        page_nums = list(range(start, end + 1))
    else:
        page_nums = list(pages)

    ocr = PaddleOCR(
        lang=lang,
        device="gpu",
        text_detection_model_dir=det_model_dir,
        text_recognition_model_dir=rec_model_dir,
        textline_orientation_model_dir=textline_ori_model_dir,
        use_textline_orientation=use_textline_orientation,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    results = {}
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for p in page_nums:
        if p < 0 or p >= len(doc):
            continue
        pix = doc[p].get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = img[:, :, :3]

        raw = ocr.predict(img)
        page_res = raw[0] if raw else None
        text = "\n".join(page_res["rec_texts"]) if page_res else ""
        results[p] = text

    doc.close()
    return results

def ocr_docling(pdf_path: str, start_page: int | None = None, end_page: int | None = None) -> str:
    """
    Convert a single page or range of pages from a PDF to markdown using docling.

    Args:
        pdf_path: path to the PDF file
        start_page: 1-indexed starting page
        end_page: 1-indexed ending page (inclusive). If None, only start_page is converted.

    Returns:
        Markdown string of the specified page range.
    """
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat

    if end_page is None:
        end_page = start_page

    if start_page is None:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
    else:
        converter = DocumentConverter()
        result = converter.convert(pdf_path, page_range=(start_page, end_page))

    return result.document.export_to_markdown()

#**
def chunk_report(pdf_fp:str, targets:list[str]|None = None, store_chuncks_locally:bool = False, store_fp:str|None = None):
    """
    This function takes the pdf and creates a document hiearchy for each of the following 3 types of reports:
    1. Single-study reports
    2. Multi-study reports (submission package or simple multi-study report)

    By utilizing the following approach:
    page-by-page analysis to identify study boundaries, where a 'new section score' is defined by the variables:
    Title_page_like,
    Study_preface_like,
    Repeat_Summary,
    Reset_pages,
    Reset_study_id,
    New_substance
    
    returns a list of Chunck Objects.
    Optional. Stores the chunck results as separate pdfs to a local filepath (folder).
    """
    import pymupdf
    import re
    from collections import Counter

    class Chunk:
        def __init__(self,page_start:int,page_end:int,title:str|None): # page numbers are 1-based.
            self.page_start = page_start
            self.page_end = page_end
            self.title = title
            self.page_range = [n for n in range(page_start+1,page_end+1)]

    def title_page_likeliness(text:str):
        """
        Returns:
        score: float between 0 and 1
        features: dictionary of feature values
        """
        TITLE_KEYWORDS = {
            "final report",
            "study",
            "study report",
            "test substance",
            "substance",
            "project",
            "project number",
            "study number",
            "protocol",
            "sponsor",
            "laboratory",
            "performed for",
            "prepared for",
            "quality assurance",
            "glp",
            "good laboratory practice",
            "confidential",
            "report",
            "author",
            "study director",
        }

        SECTION_HEADINGS = {
            "summary",
            "introduction",
            "materials",
            "methods",
            "results",
            "discussion",
            "conclusion",
            "references",
            "appendix",
        }

        if not page_text.strip():
            return 0.0, {}

        lines = [l.strip() for l in page_text.splitlines() if l.strip()]
        lower = page_text.lower()

        score = 0
        features = {}

        # 1. Low amount of text

        word_count = len(re.findall(r"\b\w+\b", page_text))
        features["word_count"] = word_count

        if word_count < 150:
            score += 2
        elif word_count < 250:
            score += 1

        # 2. Few long paragraphs

        long_lines = sum(len(l.split()) > 15 for l in lines)
        features["long_lines"] = long_lines

        if long_lines <= 2:
            score += 2

        # 3. Many short centered-looking lines

        short_lines = sum(len(l.split()) <= 6 for l in lines)
        features["short_lines"] = short_lines

        if short_lines >= len(lines) * 0.5:
            score += 2

        # 4. Title keywords

        keyword_hits = []

        for keyword in TITLE_KEYWORDS:
            if keyword in lower:
                keyword_hits.append(keyword)

        features["title_keywords"] = keyword_hits

        score += min(len(keyword_hits), 5)

        # 5. Penalize section headings

        section_hits = []

        for heading in SECTION_HEADINGS:
            if heading in lower:
                section_hits.append(heading)

        features["section_headings"] = section_hits

        score -= len(section_hits)

        # 6. Penalize lots of numeric data

        numbers = re.findall(r"\d+\.\d+|\d+", page_text)

        features["numbers"] = len(numbers)

        if len(numbers) > 50:
            score -= 2

        # 7. Peanlize TOC-like pages
        if is_toc(text):
            score -= 4
            features["toc_like"] = True
        else:
            features["toc_like"] = False

        # Normalize

        probability = max(0, min(score / 12, 1))

        return probability, features

    def study_preface_likeliness(text:str):
        """
        Returns:
        study_preface_score: 0 - 1
        features: {}
        """
        study_preface_keywords = [
            "study director",
            "sponsor",
            "glp compliance",
            "quality assurance"
        ]

        if not page_text.strip():
            return 0.0, {}

        lines = [l.strip() for l in page_text.splitlines() if l.strip()]
        lower = page_text.lower()

        score = 0
        features = {}

        # Count words
        if len(text.split(' ')) < 150:
            score += 2
            features['word count'] = 'Low'
        elif len(text.split(' ')) < 200:
            score += 1
            features['word count'] = 'Medium-Low'
        else:
            features['word count'] = 'Medium to High'
       
        # Count occurance of key words
        if any(w.lower() in text.strip().lower() for w in study_preface_keywords):
            score += 4
            features['key words present'] = True
        else:
            features['key words present'] = False
        
        # Detect study-like section headers
        section_hot_words = ["introduction","1ntroduction","summary","sumnary","abstract","test substance","test material","procedure","results","discussion","references","methods","cell line","harvest","staining","coding"]
        sections = detect_sections(text=text)
        num_hits = 0
        for s in sections:
            if any(w in s.title.lower() for w in section_hot_words):
                num_hits += 1
        
        if num_hits < 1: # no study-like titles
            score += 2
        
        # penalize study-like titles
        if num_hits > 1:
            score -= 2

        features['Num_study_like_titles'] = num_hits

        # penalize TOC
        if is_toc(text):
            score -= 4
            features['is_toc'] = True
        else:
            features['is_toc'] = False

        probability = max(0, min(score / 8, 1))

        return probability, features


    pages = [] # cleaned text
    with pymupdf.open(pdf_fp) as doc:
        ps = [p.get_text() for p in doc]
        for p in ps:
            pages.append(clean_pymupdf_text(p))
    
    page_boundary_scores = {} # page number, score
    
    for pdx in range(len(pages)):
        p = pages[pdx]
        if is_toc(p):
            continue
        
        title_page_score, _ = title_page_likeliness(p)
        study_preface_score, _ = study_preface_likeliness(p)
