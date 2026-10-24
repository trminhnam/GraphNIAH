grammar acl;

// ──────────────────────────────────────────────
// Parser Rules
// ──────────────────────────────────────────────

reference
    : authors DOT year DOT body EOF
    ;

// Author list: comma-separated, last author preceded by "and",
// or ending with "et al." / "and et al." / "and N others".
//
// ACL natbib format uses first-name-first ordering:
//   FirstName LastName, FirstName LastName, and FirstName LastName.
authors
    : authorName (COMMA authorName)* COMMA AND authorName
    | authorName (COMMA authorName)* COMMA AND NUMBER WORD
    | authorName (COMMA authorName)* COMMA AND ET_AL
    | authorName (COMMA authorName)* COMMA ET_AL
    | authorName AND authorName
    | authorName ET_AL
    | authorName
    ;

authorName
    : nameWord+
    ;

nameWord
    : WORD
    | INITIAL
    | NAMETITLE
    | APOS
    | DASH
    | AT
    ;

year
    : NUMBER NUMBER? WORD?
    ;

// Everything after "year." — title + optional venue.
// The Python layer splits this into title vs. venue.
body
    : bodyToken*
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
    | NAMETITLE
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

ET_AL     : 'et' [ \t\r\n]+ 'al' ;
AND       : 'and' ;
NAMETITLE : ('Jr' | 'Sr' | 'II' | 'III' | 'IV') '.' ;
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
