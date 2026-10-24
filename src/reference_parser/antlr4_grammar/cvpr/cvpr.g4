grammar cvpr;

// ──────────────────────────────────────────────
// Parser Rules
// ──────────────────────────────────────────────

reference
    : authors body EOF
    ;

// Author list: comma-separated, last author preceded by "and",
// or ending with "et al."
//
// CVPR abbrvnat uses first-name-first ordering (same as ACL):
//   FirstName LastName, FirstName LastName, and FirstName LastName.
// Year is NOT placed here — it appears at the end of the body.
//
// The author-body separator DOT is consumed here for non-et-al
// alternatives.  For et-al alternatives the period is already
// inside the ET_AL token ('et al.'), so no extra DOT is needed.
authors
    : authorName (COMMA authorName)* COMMA AND authorName DOT
    | authorName (COMMA authorName)* COMMA ET_AL
    | authorName AND authorName DOT
    | authorName ET_AL
    | authorName DOT
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
    ;

// Everything after "authors." — title + venue + year.
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
