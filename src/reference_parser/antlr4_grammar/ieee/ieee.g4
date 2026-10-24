grammar ieee;

// ──────────────────────────────────────────────
// Parser Rules
// ──────────────────────────────────────────────

// IEEE reference structures produced by IEEEtran.bst:
//
//   1. @article / @inproceedings / @incollection / @techreport / @thesis:
//      F. LastName, G. LastName, and H. LastName, "Title," venue, vol. X, no. Y, pp. Z, Year.
//
//   2. @book / @manual / @inbook (title is italicized — no quotes after PDF extraction):
//      F. LastName and G. LastName, Book Title. Location: Publisher, Year.
//
//   3. @electronic / @misc (period separators, author may be absent):
//      F. LastName. "Title." Source. [Online]. Available: URL
//      "Title," Source. [Online]. Available: URL          (authorless)
//
//   4. Corporate / organizational author (no initials):
//      Organization Name, "Title," venue, Year.
//
// Distinctive IEEE features that keep this grammar strict:
//   - Personal author names are initials-first: "F. von LastName" (starts with INITIAL)
//   - Corporate authors start with WORD/NUMBER — disjoint from personal authors
//   - Authorless references start with a quoted title (DQUOTE / APOS)
reference
    : authors separator title body EOF            // quoted title (article, inproceedings, etc.)
    | authors separator body EOF                  // unquoted title (book, manual — italics lost)
    | corporateAuthor separator title body EOF    // corporate author + quoted title
    | corporateAuthor separator body EOF          // corporate author + unquoted / venue-only
    | title body EOF                              // authorless with quoted title
    | title EOF                                   // authorless, title only
    ;

// Separator between author/corporate block and what follows.
// IEEEtran.bst normally uses commas; the @electronic type uses periods.
separator
    : COMMA
    | DOT
    ;

// ── Author list ───────────────────────────────
// IEEE uses comma-separated, initials-first names; "and" precedes the last author.
// "et al." may appear with or without a preceding comma.
authors
    : authorName (COMMA authorName)* COMMA AND authorName
    | authorName (COMMA authorName)* COMMA ET_AL
    | authorName (COMMA authorName)* ET_AL
    | authorName AND authorName
    | authorName
    ;

// ── Corporate / organizational author ─────────
// IEEE supports corporate authors (e.g., "Federal Communications Commission",
// "IEEE Standards Association", "3GPP").  These do NOT follow the initials-first
// personal name format — they begin with a WORD or NUMBER token, which is
// disjoint from the INITIAL token that starts personal authorName.
corporateAuthor
    : (WORD | NUMBER) (WORD | DASH | AMP | APOS | NUMBER | AND | SEMICOLON | COLON | LPAREN | RPAREN | PLUS)*
    ;

// IEEE name format: F. [von] LastName  (one or more initials, optional particles, last name)
// Examples: "J. Smith"  "A. B. Jones"  "Z.-H. Deng"  "R. von Mises"  "M. O'Brien"
authorName
    : initials nameWord+
    ;

// One or more initials, optionally joined by hyphens (e.g. "Z.-H." or "A. B.")
initials
    : INITIAL (DASH? INITIAL)*
    ;

nameWord
    : WORD
    | APOS
    | DASH
    ;

// ── Title ─────────────────────────────────────
// IEEE encloses article/paper titles in double quotes; PDF extractors sometimes
// produce curly single quotes instead.  The trailing separator comma is typically
// included inside the closing quote by the IEEEtran.bst formatter ("Title,").
title
    : DQUOTE titleTokenDq* DQUOTE
    | APOS   titleTokenApos* APOS
    ;

// Inside a double-quoted title: every token except DQUOTE (which closes the title).
titleTokenDq
    : WORD | AND | NUMBER | COMMA | COLON | DASH | LPAREN | RPAREN
    | DOT  | QUESTION | EXCL | APOS | SLASH | AMP | SEMICOLON | INITIAL
    | HASH | TILDE | UNDERSCORE | STAR | AT | ET_AL
    | LBRACKET | RBRACKET | PERCENT | PLUS | EQUALS
    ;

// Inside a single-quoted title: every token except APOS (which closes the title).
titleTokenApos
    : WORD | AND | NUMBER | COMMA | COLON | DASH | LPAREN | RPAREN
    | DOT  | QUESTION | EXCL | DQUOTE | SLASH | AMP | SEMICOLON | INITIAL
    | HASH | TILDE | UNDERSCORE | STAR | AT | ET_AL
    | LBRACKET | RBRACKET | PERCENT | PLUS | EQUALS
    ;

// ── Body ──────────────────────────────────────
// Everything after the title (or after the separator for unquoted-title
// references): journal/conference name, volume, number, pages, year, URLs, etc.
body
    : bodyToken+
    ;

bodyToken
    : WORD | AND | NUMBER | COMMA | COLON | DASH | LPAREN | RPAREN
    | DOT  | QUESTION | EXCL | APOS | SLASH | AMP | SEMICOLON | INITIAL
    | HASH | TILDE | UNDERSCORE | STAR | DQUOTE | AT | ET_AL
    | LBRACKET | RBRACKET | PERCENT | PLUS | EQUALS
    ;

// ──────────────────────────────────────────────
// Lexer Rules (order matters – most specific first)
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
LBRACKET  : '[' ;
RBRACKET  : ']' ;
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
PERCENT   : '%' ;
PLUS      : '+' ;
EQUALS    : '=' ;
NUMBER    : [0-9]+ ;
WORD      : [a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F]+ ;
WS        : [ \t\r\n]+ -> skip ;
