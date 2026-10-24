grammar iclr;

// ──────────────────────────────────────────────
// Parser Rules
// ──────────────────────────────────────────────

reference
    : authors body EOF
    ;

// Author list: comma-separated, last author preceded by "and",
// or ending with "et al."
//
// ICLR natbib style uses first-name-first ordering:
//   FirstName LastName, FirstName LastName, and FirstName LastName.
// Year is NOT fixed right after the author list; it appears in body.
//
// DOT is absorbed into each non-ET_AL alternative because BibTeX's
// add.period$ places a single '.' after the author block — but when
// the last author is "et al.", the ET_AL lexer token already includes
// that period, so no separate DOT follows.
authors
    : authorName (COMMA authorName)* COMMA AND authorName DOT
    | authorName (COMMA authorName)* COMMA ET_AL
    | authorName ET_AL
    | authorName AND authorName DOT
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
    | SUFFIX
    ;

// Everything after "authors." — title + venue + year (+ optional URL/DOI blocks).
// Must start with a WORD or APOS (ICLR natbib style allows titles that begin
// with a quoted word, e.g. 'Infodemic').  Starting with DQUOTE is still rejected
// to avoid matching IEEE-style references whose titles are wrapped in double
// quotes.  The Python layer extracts year and splits title vs. venue.
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
// Lexer Rules (order matters – most specific first)
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
