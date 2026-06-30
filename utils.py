# The new utilities file to support version 9.0 onwards
# Last edited: June 05, 2026

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

    _PAGE_NUMBER_RE = re.compile(
        r'''^\s*
        (
            -\s*\d{1,4}\s*-        |   # -6-
            \(\s*\d{1,4}\s*\)      |   # (6)
            \[\s*\d{1,4}\s*\]      |   # [6]
            Page\s+\d{1,4}         |   # Page 6
            \d{1,4}\s*/\s*\d{1,4}     # 6/14  (current/total)
        )
        \s*$''',
        re.IGNORECASE | re.VERBOSE,
    )

    _PAGE_OF_RE = re.compile(
    r'\bpage\s+\d{1,4}\s+of\s+\d{1,4}\b',
    re.IGNORECASE
    )

    def _isolate_page_of(text: str) -> str:
        # Surround matches with newlines so they become standalone lines
        return _PAGE_OF_RE.sub(r'\n\g<0>\n', text)
    def _remove_page_numbers(lines: list[str]) -> list[str]:
        matches = [i for i, ln in enumerate(lines) if _PAGE_NUMBER_RE.match(ln)]
        if len(matches) < 2:
            return lines
        remove = set(matches)
        return [ln for i, ln in enumerate(lines) if i not in remove]


    # ------------------------------------------------------------------ #
    # Stage 2 — Repeated header / footer blocks
    # ------------------------------------------------------------------ #

    def _remove_repeated_header_footer_blocks(
        lines: list[str],
        min_repeats: int = 4,
        block_radius: int = 3,
    ) -> list[str]:
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

    lines = _remove_page_numbers(lines)
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

def is_toc(text: str, min_entries: int = 4, min_ratio: float = 0.35) -> bool:
    """
    Return True if *text* looks like a Table of Contents page.
 
    Parameters
    ----------
    text        : raw page string to test
    min_entries : minimum number of TOC-like lines required
    min_ratio   : minimum fraction of non-blank lines that must be TOC-like
    """
    import re
    # Matches a (possibly noisy) numbered section at the start of a line, e.g.:
    #   "1.", "3.4", "3 .4", "_ 3 .", "a` 4 .1", "3.10", "3 .11"
    _SECTION_LINE_RE = re.compile(
        r"""
        ^                           # start of line
        [\s`'_~•·\-–—]*            # leading noise chars (underscores, backticks…)
        [a-zA-Z]{0,2}              # optional stray alpha prefix (e.g. "a`")
        [\s`'_~•·\-–—]*            # more noise
        \d+                         # first number segment
        (?:[\s]*\.[\s]*\d+)*        # optional .N subsections (spaces around dots ok)
        [\s]*\.?\s                  # trailing dot/space separator
        """,
        re.VERBOSE | re.MULTILINE,
    )
    
    # ALL-CAPS unnumbered headings, e.g. "SUMMARY OF RESULTS"
    _CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s\-/&(),:.]{4,}$", re.MULTILINE)

    non_blank_lines = [l for l in text.splitlines() if l.strip()]
    if not non_blank_lines:
        return False
 
    numbered = len(_SECTION_LINE_RE.findall(text))
    caps     = len(_CAPS_HEADING_RE.findall(text))
    matched  = numbered + caps
 
    ratio = matched / len(non_blank_lines)
    return matched >= min_entries and ratio >= min_ratio

def detect_sections(pdf_fp,target_titles:list|None=None,searching:bool=False):
    """
    Inputs:
    pdf_fp: str
    searching:bool, if true, must provide a list of target titles to search
    target_titles: list of string

    Outputs:
    Sections - List of Section Objects
    """
    import re
    import fitz
    import unicodedata

    sections = []
    debug = []

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
            return sections, debug
        
        elif searching == True:
            if target_titles:
                # Search for the relevant sections and return them
                target_sections = []
                for target in target_titles:
                    for section in sections:
                        if (target.strip().lower() in section.title.strip().lower()) and (section.content not in [t.content for t in target_sections]):
                            target_sections.append(section)

                if len(target_sections)==0:
                    return target_sections, debug

                organized_target_sections = []

                for section in target_sections:
                    page_num = section.page_num

                    # First and foremost detect if the page is a TOC
                    if is_toc(section.content):
                        continue # do not add to organized_target_sections if page is a TOC
                    
                    # Second detect if the page fully contains the section (includes multiple sections)
                    section_hot_words = ["introduction","1ntroduction","summary","sumnary","abstract","test substance","test material","procedure","results","discussion","references","histopathology","hematology", "mortality", "necropsy","methods","cell line","preliminary","harvest","culture","staining","study","coding"]
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
                                    if match2 > match:
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

                        else: # Only the target section title exists on this page
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

                    for line in clean_next_page.split('\n'): # search line by line
                        line = line.strip()
                        if ( len(line.split()) > 10 ): # first sentence
                            debug.append(line)
                            if line[0].islower(): # first sentence is lowercase
                                debug.append(f"IS LOWER: {page_num+1}")
                                if (len(next_matches)+len(next_matches))>0: # More than 1 title exists on the next page
                                    non_target_titles2 = []
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
                                if (len(next_matches)>0) or (len(next_matches2)):
                                    if any(t.strip().lower() in excluded_hot_words for t in next_matches) or any(t.strip().lower() in excluded_hot_words for t in next_matches2): # next page contains a new section
                                        break

                                    else: # next page does not begin a new section
                                        if (page_num+1) not in [o.page_num for o in organized_target_sections]:
                                            next_page_sec = Section(
                                                page_num = page_num+1,
                                                content = clean_next_page,
                                                title = str(f"{section.title}; continued page")
                                            )
                                            organized_target_sections.append(next_page_sec)
                                            break
                                        else: # Page already included in organized target sections
                                            break

                return organized_target_sections,debug

