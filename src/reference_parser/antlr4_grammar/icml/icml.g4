grammar icml;

// ──────────────────────────────────────────────
// Parser Rules
// ──────────────────────────────────────────────

reference
    : authors DOT? body EOF
    ;

// Author list strictly following icml2025.bst output format.
//
// The .bst format string {vv~}{ll}{, jj}{, f.} produces:
//   LastName, F.  (personal author — "formal" format with initials)
//   OrgName       (organizational author — "bare" without initials)
//
// Strictness rules derived from icml2025.bst:
// ─────────────────────────────────────────────
// 1. Multi-author lists MUST contain at least one "authorFormal"
//    (LastName, Initials) to distinguish ICML from other styles
//    (e.g. ACL first-name-first: "John Smith and Alice Jones").
// 2. Oxford comma appears ONLY with 3+ authors (the .bst guards
//    with `numnames #2 >`), so the 3+-author alt uses `+`.
// 3. For exactly 2 authors there is NO Oxford comma:
//      LastName, F. and SecondLastName, F.
// 4. "et al." replaces {and others} in BibTeX:
//    - 2 names (1 real + others): no comma before et al.
//        LastName, F. et al.
//    - 3+ names (2+ real + others): Oxford comma before et al.
//        LastName, F., SecondLastName, F., et al.
// 5. Junior suffixes ({, jj} → "Jr.", "III") create inherent
//    ambiguity with multi-author comma lists, so they are handled
//    by the Python fallback parser rather than in the grammar.
//
// Year is NOT placed here — it appears at the end of the body.
authors
    : authorFormal (COMMA authorName)+ COMMA AND authorName   // 3+ authors (Oxford comma)
    | authorFormal AND authorName                               // 2 authors (no Oxford comma)
    | authorName AND authorFormal                               // 2 authors, bare first
    | authorFormal (COMMA authorName)+ COMMA ET_AL              // 3+ (incl. others) with et al.
    | authorFormal ET_AL                                        // 1 real author + et al. (no comma)
    | authorName                                                // single author
    ;

// Canonical ICML personal-author format: LastName, F. [M.]
// Requires a comma followed by abbreviated initials.
authorFormal
    : nameParts COMMA initials
    ;

// General author: either the formal format or a bare name (orgs).
authorName
    : nameParts COMMA initials
    | nameParts
    ;

nameParts
    : nameWord+
    ;

nameWord
    : WORD
    | APOS
    | DASH
    ;

initials
    : initial (DASH? initial)*
    ;

initial
    : INITIAL
    ;

// Everything after authors — title + venue + year.
// The Python layer extracts the year and splits title vs. venue.
body
    : bodyToken+
    ;

bodyToken
    : WORD
    | AND
    | NUMBER
    | COMMA
    | COLON
    | DASH
    | LPAREN
    | RPAREN
    | DOT
    | QUESTION
    | EXCL
    | APOS
    | SLASH
    | AMP
    | SEMICOLON
    | INITIAL
    | HASH
    | TILDE
    | UNDERSCORE
    | STAR
    | DQUOTE
    | AT
    | ET_AL
    ;

// ──────────────────────────────────────────────
// Lexer Rules  (order matters – most specific first)
// ──────────────────────────────────────────────

ET_AL     : 'et' [ \t\r\n]+ 'al' '.' ;
AND       : 'and' ;
INITIAL   : [A-Z\u00C0-\u00D6\u00D8-\u00DE] '.' ;
SEMICOLON : ';' ;
DOT       : '.' ;
COMMA     : ',' ;
COLON     : ':' ;
DASH      : '-' | '\u2013' | '\u2014' ;
LPAREN    : '(' ;
RPAREN    : ')' ;
QUESTION  : '?' ;
EXCL      : '!' ;
APOS      : '\'' | '\u2019' | '\u2018' ;
SLASH     : '/' ;
AMP       : '&' ;
HASH      : '#' ;
TILDE     : '~' ;
UNDERSCORE: '_' ;
STAR      : '*' ;
DQUOTE    : '"' | '\u201C' | '\u201D' ;
AT        : '@' ;
NUMBER    : [0-9]+ ;
WORD      : [a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F]+ ;
WS        : [ \t\r\n]+ -> skip ;
