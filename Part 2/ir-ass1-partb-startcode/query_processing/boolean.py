from typing import Set, List, Tuple
from index.access import get_posting_list
from index.io import load as _load_pkg

OPS = {"AND", "OR", "NOT"}
_PRECEDENCE = {"OR": 1, "AND": 2, "NOT": 3}
_RIGHT_ASSOC = {"NOT"}  # unary NOT

def _tokenize_boolean(query: str) -> List[str]:
    """
    Tokenize while:
      - keeping quoted phrases as a single token: '"machine learning"'
      - separating parentheses () as tokens
      - keeping AND/OR/NOT as uppercase operator tokens
      - everything else becomes a single term token
    Validations:
      - matched quotes; non-empty phrases; phrase length ≤ 3
      - balanced parentheses
      - operator placement:
          • disallow ANY leading operator (including NOT)
          • disallow trailing operator
          • disallow operator immediately after '(' (including NOT)
          • allow (AND|OR) followed by one or more NOTs before an operand
          • allow NOT chains only when not at the start (e.g., A AND NOT NOT B)
      - forbid two operators in a row except the allowed NOT-chain forms above
      - forbid adjacent operands without an operator
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")

    tokens: List[str] = []
    buf = []
    i, n = 0, len(query)

    def flush_buf():
        nonlocal buf, tokens
        if buf:
            tokens.append("".join(buf))
            buf = []

    while i < n:
        ch = query[i]
        if ch == '"':
            flush_buf()
            j = query.find('"', i + 1)
            if j == -1:
                raise ValueError("Malformed: unmatched quotes.")
            phrase = query[i + 1 : j]
            if phrase.strip() == "":
                raise ValueError('Malformed: empty phrase "".')
            words = phrase.split()
            if len(words) > 3:
                raise ValueError("Malformed: phrase longer than 3 words.")
            tokens.append(f'"{phrase}"')
            i = j + 1
        elif ch in ("(", ")"):
            flush_buf()
            tokens.append(ch)
            i += 1
        elif ch.isspace():
            flush_buf()
            i += 1
        else:
            buf.append(ch)
            i += 1
    flush_buf()

    # Balanced parentheses
    if tokens.count("(") != tokens.count(")"):
        raise ValueError("Malformed: unbalanced parentheses.")

    # Operator placement checks
    for i, tok in enumerate(tokens):
        if tok in OPS:
            # REGRESSION-SUITE RULE: disallow ANY leading operator (including NOT)
            if i == 0:
                raise ValueError("Malformed: leading operator.")
            if i == len(tokens) - 1:
                raise ValueError("Malformed: trailing operator.")

            # two operators in a row
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt in OPS:
                # Allow (AND|OR) followed by a chain of NOTs before an operand:  A AND NOT NOT term
                if tok in ("AND", "OR") and nxt == "NOT":
                    j = i + 2
                    while j < len(tokens) and tokens[j] == "NOT":
                        j += 1
                    if j >= len(tokens) or tokens[j] in OPS or tokens[j] == ")":
                        raise ValueError("Malformed: NOT missing operand.")
                # Allow NOT followed by NOT (chain) only when not at the start:  ... NOT NOT term
                elif tok == "NOT" and nxt == "NOT":
                    j = i + 1
                    while j < len(tokens) and tokens[j] == "NOT":
                        j += 1
                    if j >= len(tokens) or tokens[j] in OPS or tokens[j] == ")":
                        raise ValueError("Malformed: NOT missing operand.")
                else:
                    raise ValueError("Malformed: two operators in a row.")

        if tok == "(":
            # REGRESSION-SUITE RULE: next cannot be ')' or ANY operator (including NOT)
            if i + 1 < len(tokens) and (tokens[i + 1] in OPS or tokens[i + 1] == ")"):
                raise ValueError("Malformed: empty () or operator right after '('.")

        if tok == ")":
            # prev cannot be '(' or operator
            if i - 1 >= 0 and (tokens[i - 1] in OPS or tokens[i - 1] == "("):
                raise ValueError("Malformed: empty () or operator before ')'.")

    # Adjacent operands without operator, e.g., term "phrase" or ) term, term (
    def is_operand(t: str) -> bool:
        if t in OPS or t in ("(", ")"):
            return False
        return True

    for a, b in zip(tokens, tokens[1:]):
        if (is_operand(a) and is_operand(b)) or (a == ")" and is_operand(b)) or (is_operand(a) and b == "("):
            raise ValueError("Malformed: adjacent operands without operator (insert AND/OR).")

    return tokens

def _to_rpn(tokens: List[str]) -> List[str]:
    """Shunting-yard to RPN with precedence NOT > AND > OR (NOT is unary)."""
    out: List[str] = []
    st: List[str] = []

    for tok in tokens:
        if tok in OPS:
            while st and st[-1] in OPS:
                top = st[-1]
                if (_PRECEDENCE[top] > _PRECEDENCE[tok]) or (
                    _PRECEDENCE[top] == _PRECEDENCE[tok] and tok not in _RIGHT_ASSOC
                ):
                    out.append(st.pop())
                else:
                    break
            st.append(tok)
        elif tok == "(":
            st.append(tok)
        elif tok == ")":
            while st and st[-1] != "(":
                out.append(st.pop())
            if not st:
                raise ValueError("Malformed: unbalanced parentheses.")
            st.pop()  # pop '('
        else:
            out.append(tok)

    while st:
        op = st.pop()
        if op in ("(", ")"):
            raise ValueError("Malformed: unbalanced parentheses.")
        out.append(op)

    return out

def _phrase_to_tuple(phrase_literal: str) -> Tuple[str, ...]:
    # '"a b c"' -> ('a', 'b', 'c')
    assert phrase_literal.startswith('"') and phrase_literal.endswith('"')
    inside = phrase_literal[1:-1].strip()
    words = inside.split()
    if len(words) == 0 or len(words) > 3:
        raise ValueError("Malformed phrase.")
    return tuple(words)

def _eval_rpn(rpn: List[str], index_path: str) -> Set[int]:
    """Evaluate RPN using posting lists; NOT = universe \\ set."""
    # Universe for NOT
    meta = _load_pkg(index_path).get("__META__", {})
    universe: Set[int] = set(meta.get("doc_lengths", {}).keys())

    def postings(tok: str) -> Set[int]:
        if tok.startswith('"') and tok.endswith('"'):
            tup = _phrase_to_tuple(tok)
            return set(get_posting_list(tup, index_path))
        else:
            return set(get_posting_list(tok, index_path))

    st: List[Set[int]] = []
    for tok in rpn:
        if tok in OPS:
            if tok == "NOT":
                if not st:
                    raise ValueError("Malformed: NOT missing operand.")
                a = st.pop()
                st.append(universe - a)
            else:
                # binary op
                if len(st) < 2:
                    raise ValueError("Malformed: binary operator missing operands.")
                b = st.pop()
                a = st.pop()
                st.append(a & b if tok == "AND" else a | b)
        else:
            st.append(postings(tok))

    if len(st) != 1:
        raise ValueError("Malformed expression.")
    return st[0]

def process_boolean_query(query: str, index_path: str) -> Set[int]:
    """
    Process Boolean queries with AND/OR/NOT, parentheses, and quoted phrases.
    Precedence: NOT > AND > OR.
    """
    tokens = _tokenize_boolean(query)
    rpn = _to_rpn(tokens)
    return _eval_rpn(rpn, index_path)
