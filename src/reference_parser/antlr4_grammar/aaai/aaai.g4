grammar aaai;

// ──────────────────────────────────────────────
// Parser Rules
// ──────────────────────────────────────────────

reference
    : authors DOT? year body EOF
    ;

// Author list: semicolon-separated (or comma-separated), last author preceded by "and",
// or ending with "et al."
authors
    : authorName (SEMICOLON authorName)* SEMICOLON AND authorName SEMICOLON etAl
    | authorName (SEMICOLON authorName)* SEMICOLON AND authorName
    | authorName (SEMICOLON authorName)* SEMICOLON etAl
    | authorName (COMMA authorName)* COMMA AND authorName
    | authorName
    ;

etAl
    : WORD WORD DOT     // "et" "al."
    ;

authorName
    : nameParts COMMA initials
    | nameParts
    ;

nameParts
    : nameWord+
    ;

nameWord
    : WORD
    | NUMBER
    | APOS
    | DASH
    ;

initials
    : initial (DASH? initial)*
    ;

initial
    : INITIAL
    ;

// Year always ends with a period; the suffix letter (e.g. '2024a.') may be
// tokenized as INITIAL (single lowercase letter + dot) when lowercase INITIAL
// support is active, so both forms are accepted explicitly.
year
    : NUMBER INITIAL      // 2024a.  – suffix + dot absorbed as one INITIAL token
    | NUMBER WORD? DOT    // 2024.  or (rarely) 2024X. with uppercase suffix
    ;

// Everything after "year." — title + optional venue.
// The Python layer splits this into title vs. venue.
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
    ;

// ──────────────────────────────────────────────
// Lexer Rules  (order matters – most specific first)
// ──────────────────────────────────────────────

AND       : 'and' ;
INITIAL   : [A-Za-z\u00C0-\u00D6\u00D8-\u00DE\u00E0-\u00F6\u00F8-\u00FF] '.' ;
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
