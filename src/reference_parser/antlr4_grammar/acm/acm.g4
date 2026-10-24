grammar acm;

// ──────────────────────────────────────────────
// Parser Rules
// ──────────────────────────────────────────────

// ACM references normally follow: Authors. Year. Body
// Some entries (corporate/anonymous) omit the author block entirely.
// Some entries omit the year (preprints, undated corporate docs).
reference
    : authors DOT year DOT body EOF
    | authors DOT body EOF
    | year DOT body EOF
    ;

// Author list: comma-separated, last author preceded by "and",
// ending with "et al.", "and others", or a plain comma-only list.
//
// ACM style uses first-name-first ordering:
//   FirstName [Initials] LastName, ..., and FirstName LastName
// Year appears immediately after the author-list dot.
//
// Alternatives ordered most-specific-first to avoid ambiguity.
authors
    : authorName (COMMA authorName)* COMMA AND authorName AND_OTHERS
    | authorName (COMMA authorName)* COMMA AND authorName
    | authorName (COMMA authorName)* COMMA AND_OTHERS
    | authorName (COMMA authorName)* COMMA ET_AL
    | authorName AND authorName AND_OTHERS
    | authorName AND authorName
    | authorName (COMMA authorName)+
    | authorName
    ;

authorName
    : nameWord+
    ;

nameWord
    : WORD
    | INITIAL
    | NUMBER
    | APOS
    | DASH
    | LPAREN
    | RPAREN
    ;

// Year: numeric (e.g. "2023", "2023a"), "n.d." / "[n.d.]",
// or a publication-status word like "Forthcoming".
// ND is a dedicated lexer token matching exactly "n.d" (without
// trailing dot); the reference rule's mandatory DOT consumes it.
year
    : NUMBER WORD?
    | ND
    | LBRACKET ND DOT RBRACKET
    | FORTHCOMING
    ;

// Everything after "authors. year." — title + venue info.
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
    | ET_AL
    | AND_OTHERS
    | LBRACKET
    | RBRACKET
    | ND
    | FORTHCOMING
    ;

// ──────────────────────────────────────────────
// Lexer Rules (order matters – most specific first)
// ──────────────────────────────────────────────

// FORTHCOMING matches the publication-status word used in place of a year.
// It must precede WORD so the lexer captures it as a single token.
FORTHCOMING : [Ff] 'orthcoming' ;
// ET_AL deliberately omits the trailing dot so the reference rule's
// mandatory DOT after "authors" consumes the "." in "et al. YEAR."
// ND matches exactly "n.d" (the literal year placeholder for "no date").
// It must precede WORD and DOT so the lexer captures it as a single token.
ND         : 'n' '.' 'd' ;
ET_AL      : 'et' [ \t\r\n]+ 'al' ;
// AND_OTHERS must appear before AND so that "and others" is matched
// as a single token rather than AND + WORD("others").
AND_OTHERS : 'and' [ \t\r\n]+ 'others' ;
AND        : 'and' ;
INITIAL    : [A-Z\u00C0-\u00D6\u00D8-\u00DE] '.' ;
SEMICOLON  : ';' ;
DOT        : '.' ;
COMMA      : ',' ;
COLON      : ':' ;
DASH       : '-' | '\u2013' | '\u2014' ;
LPAREN     : '(' ;
RPAREN     : ')' ;
LBRACKET   : '[' ;
RBRACKET   : ']' ;
QUESTION   : '?' ;
EXCL       : '!' ;
APOS       : '\'' | '\u2019' | '\u2018' ;
SLASH      : '/' ;
AMP        : '&' ;
HASH       : '#' ;
TILDE      : '~' ;
UNDERSCORE : '_' ;
STAR       : '*' ;
DQUOTE     : '"' | '\u201C' | '\u201D' ;
AT         : '@' ;
NUMBER     : [0-9]+ ;
WORD       : [a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F]+ ;
WS         : [ \t\r\n]+ -> skip ;
