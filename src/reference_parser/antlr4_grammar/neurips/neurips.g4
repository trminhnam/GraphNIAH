grammar neurips;

// ──────────────────────────────────────────────
// Parser Rules
// ──────────────────────────────────────────────

reference
    : authors body EOF
    ;

// Author list strictly following neurips.bst output format.
//
// The .bst format string {ff~}{vv~}{ll}{, jj} produces:
//   FirstName [von] LastName[, Jr.]   (first-name-first ordering)
//
// Strictness rules derived from neurips.bst (plainnat family):
// ─────────────────────────────────────────────
// 1. Authors are in first-name-first ordering.  This is the key
//    distinguishing feature from ICML's last-name-first style
//    (which uses {vv~}{ll}{, jj}{, f.}).
// 2. Oxford comma appears ONLY with 3+ authors (the .bst guards
//    with `numnames #2 >`), so the 3+-author alt uses COMMA AND.
// 3. For exactly 2 authors there is NO Oxford comma:
//      FirstName LastName and SecondFirst SecondLast.
// 4. "et al." replaces {and others} in BibTeX:
//    - 3+ names ending with others: Oxford comma before et al.
//        First, Second, et al.
//    - 1 real + others: no comma before et al.
//        FirstName LastName et al.
// 5. Junior suffixes ({, jj} → "Jr.", "III") are handled by the
//    SUFFIX lexer token within nameWord.
//
// DOT is absorbed into each non-ET_AL alternative because BibTeX's
// add.period$ places a single '.' after the author block.  When
// the last author is "et al.", the ET_AL lexer token already includes
// that period, so no separate DOT follows.
//
// Year is NOT placed here — it appears at the end of the body.
authors
    : authorName (COMMA authorName)* COMMA AND authorName DOT  // 3+ authors (Oxford comma)
    | authorName (COMMA authorName)* COMMA ET_AL                // 3+ ending with et al.
    | authorName ET_AL                                          // 1 real author + et al.
    | authorName AND authorName DOT                             // exactly 2 (no Oxford comma)
    | authorName DOT                                            // single author
    ;

authorName
    : nameWord+
    ;

nameWord
    : WORD
    | INITIAL
    | APOS
    | DASH
    | STAR
    | SUFFIX
    ;

// Everything after "authors." — title + venue + year (+ optional URL/DOI blocks).
// Must start with a WORD or APOS (neurips.bst titles always begin with a
// capitalised word via format.title / "t" change.case$, or a quoted word).
// Starting with DQUOTE is rejected to avoid matching IEEE-style references
// whose titles are wrapped in double quotes.
// The Python layer extracts the year and splits title vs. venue.
body
    : (WORD | APOS) bodyToken*
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
    | SUFFIX
    ;

// ──────────────────────────────────────────────
// Lexer Rules  (order matters – most specific first)
// ──────────────────────────────────────────────

ET_AL     : 'et' [ \t\r\n]+ 'al' '.' ;
AND       : 'and' ;
INITIAL   : [A-Z\u00C0-\u00D6\u00D8-\u00DE] '.' ;
SUFFIX    : ('Jr' | 'Sr') '.' ;
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
