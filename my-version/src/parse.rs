//! A recursive-descent parser for the expression language.
//!
//! This is the inverse of [`crate::expr::to_string`]: it turns a formula written
//! by a person (or printed by a previous run) back into an [`Node`] tree. It is
//! the front door for every command-line feature that takes a formula —
//! `--eval-expr` today, and row filters, weight expressions, building blocks and
//! seed solutions as those issues land.
//!
//! # Grammar
//!
//! ```text
//! expr    := or_expr
//! or_expr := xor_expr ('or' xor_expr)*
//! xor_expr:= and_expr ('xor' and_expr)*
//! and_expr:= cmp_expr ('and' cmp_expr)*
//! cmp_expr:= sum (('<' | '<=' | '>' | '>=' | '==') sum)?   // non-associative
//! sum     := term (('+' | '-') term)*
//! term    := unary (('*' | '/') unary)*
//! unary   := ('-' | '+' | 'not') unary | power
//! power   := primary ('^' unary)?          // right associative
//! primary := number
//!          | meta '(' expr ')'              // structural meta operator
//!          | name '(' expr (',' expr)* ')' // function call, any arity
//!          | name                          // variable
//!          | '(' expr ')'
//! meta    := 'complexity' | 'variables' | 'constants' | 'contains'
//!          | 'contains_op'
//! ```
//!
//! Operator precedence, loosest to tightest:
//! `or` < `xor` < `and` < comparisons `< <= > >= ==` < `+`/`-` < `*`/`/` <
//! unary `-`/`+`/`not` < `^`. So `-x^2` is `-(x^2)`, `2^3^2` is `2^(3^2)` and
//! `x + 1 > y * 2` is `(x + 1) > (y * 2)`.
//!
//! `or`, `xor`, `and` are left-associative. Comparisons are **non-associative**:
//! `a < b < c` is a syntax error, so write `(a < b) and (b < c)`. `not` is a
//! prefix operator at the unary level, so `not x > 0` parses as `(not x) > 0`
//! and `(not (x > 0))` is the negated comparison. `and`/`or`/`xor`/`not` are
//! reserved words in operator position; a column with one of those names is
//! still reachable through the `xN` fallback.
//!
//! # Names
//!
//! * Variable names are matched against the `vars` slice, **case sensitively**,
//!   then against the `xN` fallback (`x0`, `x1`, …) used when a variable has no
//!   printable name. An unrecognised name is an error, not a silent `0`.
//! * Function names are the primitive names from [`Op::from_name`] (`sin`,
//!   `min`, `if`, …). A name followed by `(` is a call; a bare name is a
//!   variable, so a variable called `sin` still works on its own.
//! * `pi` and `e` are constants when they are not declared variables.
//! * The meta operators `complexity`, `variables`, `constants`, `contains` and
//!   `contains_op` are unary calls. They inspect the shape of their argument
//!   rather than evaluating it, so the call is built through
//!   [`crate::expr::node`], which caches the structural value on the node.
//!   `complexity(x * x)` prints back and re-parses to the same tree.
//!
//! # Errors
//!
//! [`ParseError`] carries the offending **1-based column**, so a caller can
//! point at the exact character that failed.

use crate::expr::{constant, input, node, Node, Op};

/// A syntax error, with the 1-based column of the offending token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub column: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (column {})", self.message, self.column)
    }
}

impl std::error::Error for ParseError {}

/// Parse `src` into an expression tree. Variable names are resolved against
/// `vars` (see the module docs for the resolution rules).
pub fn parse(src: &str, vars: &[String]) -> Result<Node, ParseError> {
    let mut p = Parser {
        src: src.as_bytes(),
        pos: 0,
        vars,
    };
    p.skip_ws();
    if p.peek().is_none() {
        return Err(p.err("empty expression"));
    }
    let n = p.expr()?;
    p.skip_ws();
    if p.pos < p.src.len() {
        return Err(p.err("unexpected trailing input"));
    }
    Ok(n)
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    vars: &'a [String],
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')) {
            self.pos += 1;
        }
    }

    fn err(&self, msg: &str) -> ParseError {
        ParseError {
            message: msg.to_string(),
            column: self.pos + 1,
        }
    }

    fn expr(&mut self) -> Result<Node, ParseError> {
        self.or_expr()
    }

    fn or_expr(&mut self) -> Result<Node, ParseError> {
        let mut lhs = self.xor_expr()?;
        while self.peek_infix_keyword() == Some(Op::Or) {
            self.skip_ws();
            self.ident();
            let rhs = self.xor_expr()?;
            lhs = node(Op::Or, vec![lhs, rhs]);
        }
        Ok(lhs)
    }

    fn xor_expr(&mut self) -> Result<Node, ParseError> {
        let mut lhs = self.and_expr()?;
        while self.peek_infix_keyword() == Some(Op::Xor) {
            self.skip_ws();
            self.ident();
            let rhs = self.and_expr()?;
            lhs = node(Op::Xor, vec![lhs, rhs]);
        }
        Ok(lhs)
    }

    fn and_expr(&mut self) -> Result<Node, ParseError> {
        let mut lhs = self.cmp_expr()?;
        while self.peek_infix_keyword() == Some(Op::And) {
            self.skip_ws();
            self.ident();
            let rhs = self.cmp_expr()?;
            lhs = node(Op::And, vec![lhs, rhs]);
        }
        Ok(lhs)
    }

    /// One (non-associative) comparison: `sum (< | <= | > | >= | ==) sum`.
    fn cmp_expr(&mut self) -> Result<Node, ParseError> {
        let lhs = self.sum()?;
        self.skip_ws();
        let op = match self.peek() {
            Some(b'<') => {
                self.pos += 1;
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    Op::LessOrEqual
                } else {
                    Op::Less
                }
            }
            Some(b'>') => {
                self.pos += 1;
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    Op::GreaterOrEqual
                } else {
                    Op::Greater
                }
            }
            Some(b'=') => {
                if self.src.get(self.pos + 1) != Some(&b'=') {
                    return Err(self.err("expected '==' (a single '=' is not an operator)"));
                }
                self.pos += 2;
                Op::Equal
            }
            _ => return Ok(lhs),
        };
        let rhs = self.sum()?;
        let n = node(op, vec![lhs, rhs]);
        // Comparisons do not chain: `a < b < c` is a syntax error.
        self.skip_ws();
        if matches!(self.peek(), Some(b'<') | Some(b'>') | Some(b'=')) {
            return Err(self.err(
                "comparisons are non-associative; combine them with 'and'/'or' or parentheses",
            ));
        }
        Ok(n)
    }

    /// If the next token (after whitespace) is one of the infix keyword
    /// operators `and`, `or`, `xor`, return it; otherwise return `None`. Does
    /// not consume anything.
    fn peek_infix_keyword(&self) -> Option<Op> {
        let mut p = self.pos;
        while matches!(
            self.src.get(p),
            Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
        ) {
            p += 1;
        }
        if !matches!(self.src.get(p), Some(c) if is_ident_start(*c)) {
            return None;
        }
        let start = p;
        while matches!(self.src.get(p), Some(c) if is_ident_char(*c)) {
            p += 1;
        }
        match std::str::from_utf8(&self.src[start..p])
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str()
        {
            "and" => Some(Op::And),
            "or" => Some(Op::Or),
            "xor" => Some(Op::Xor),
            _ => None,
        }
    }

    /// Whether the next token is exactly the identifier `kw` (case-insensitive),
    /// without consuming it.
    fn at_keyword(&self, kw: &str) -> bool {
        let mut p = self.pos;
        if !matches!(self.src.get(p), Some(c) if is_ident_start(*c)) {
            return false;
        }
        let start = p;
        while matches!(self.src.get(p), Some(c) if is_ident_char(*c)) {
            p += 1;
        }
        std::str::from_utf8(&self.src[start..p])
            .map(|w| w.eq_ignore_ascii_case(kw))
            .unwrap_or(false)
    }

    fn sum(&mut self) -> Result<Node, ParseError> {
        let mut lhs = self.term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    let rhs = self.term()?;
                    lhs = node(Op::Add, vec![lhs, rhs]);
                }
                Some(b'-') => {
                    self.pos += 1;
                    let rhs = self.term()?;
                    lhs = node(Op::Sub, vec![lhs, rhs]);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn term(&mut self) -> Result<Node, ParseError> {
        let mut lhs = self.unary()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    let rhs = self.unary()?;
                    lhs = node(Op::Mul, vec![lhs, rhs]);
                }
                Some(b'/') => {
                    self.pos += 1;
                    let rhs = self.unary()?;
                    lhs = node(Op::Div, vec![lhs, rhs]);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Node, ParseError> {
        self.skip_ws();
        match self.peek() {
            Some(b'-') => {
                self.pos += 1;
                Ok(node(Op::Neg, vec![self.unary()?]))
            }
            Some(b'+') => {
                self.pos += 1;
                self.unary()
            }
            _ => {
                if self.at_keyword("not") {
                    self.ident();
                    return Ok(node(Op::Not, vec![self.unary()?]));
                }
                self.power()
            }
        }
    }

    fn power(&mut self) -> Result<Node, ParseError> {
        let base = self.primary()?;
        self.skip_ws();
        if self.peek() == Some(b'^') {
            self.pos += 1;
            // Right associative; the exponent may itself be unary (`2^-3`).
            let exp = self.unary()?;
            return Ok(node(Op::Pow, vec![base, exp]));
        }
        Ok(base)
    }

    fn primary(&mut self) -> Result<Node, ParseError> {
        self.skip_ws();
        match self.peek() {
            Some(b'(') => {
                self.pos += 1;
                let n = self.expr()?;
                self.skip_ws();
                if self.peek() != Some(b')') {
                    return Err(self.err("expected ')'"));
                }
                self.pos += 1;
                Ok(n)
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => self.number(),
            Some(c) if is_ident_start(c) => self.name_or_call(),
            Some(_) => Err(self.err("expected a number, variable, function or '('")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn number(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        // An exponent only if it is well formed; otherwise leave the 'e' for the
        // identifier parser (so `2e` stays a number followed by a variable).
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            let save = self.pos;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    self.pos += 1;
                }
            } else {
                self.pos = save;
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        text.parse::<f64>().map(constant).map_err(|_| ParseError {
            message: format!("invalid number '{text}'"),
            column: start + 1,
        })
    }

    fn ident(&mut self) -> String {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if is_ident_char(c)) {
            self.pos += 1;
        }
        String::from_utf8_lossy(&self.src[start..self.pos]).into_owned()
    }

    fn name_or_call(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        let name = self.ident();
        self.skip_ws();
        if self.peek() != Some(b'(') {
            // A bare name: a declared variable, then the `xN` fallback, then a
            // named constant.
            if let Some(i) = self.vars.iter().position(|v| v == &name) {
                return Ok(input(i));
            }
            if let Some(i) = x_index(&name) {
                return Ok(input(i));
            }
            match name.as_str() {
                "pi" => return Ok(constant(std::f64::consts::PI)),
                "e" => return Ok(constant(std::f64::consts::E)),
                _ => {}
            }
            return Err(ParseError {
                message: format!("unknown variable '{name}'"),
                column: start + 1,
            });
        }

        // A function call.
        self.pos += 1;
        let mut args = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b')') {
            self.pos += 1;
        } else {
            loop {
                args.push(self.expr()?);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => self.pos += 1,
                    Some(b')') => {
                        self.pos += 1;
                        break;
                    }
                    _ => return Err(self.err("expected ',' or ')' in argument list")),
                }
            }
        }
        let op = match Op::from_name(&name) {
            Some(op) if op.arity() > 0 => op,
            Some(_) => {
                return Err(ParseError {
                    message: format!("'{name}' is not a function"),
                    column: start + 1,
                })
            }
            None => {
                return Err(ParseError {
                    message: format!("unknown function '{name}'"),
                    column: start + 1,
                })
            }
        };
        if op.arity() != args.len() {
            return Err(ParseError {
                message: format!(
                    "'{name}' takes {} argument{}, got {}",
                    op.arity(),
                    if op.arity() == 1 { "" } else { "s" },
                    args.len()
                ),
                column: start + 1,
            });
        }
        Ok(node(op, args))
    }
}

/// The `x0`, `x1`, … fallback used when a variable has no printable name.
fn x_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix('x')?;
    if rest.is_empty() || !rest.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::same;

    fn vars() -> Vec<String> {
        ["x", "y", "z"].iter().map(|s| s.to_string()).collect()
    }

    fn p(s: &str) -> Node {
        parse(s, &vars()).expect(s)
    }

    fn col(s: &str) -> usize {
        parse(s, &vars()).expect_err(s).column
    }

    #[test]
    fn parses_literals_and_variables() {
        assert!(same(&p("42"), &constant(42.0)));
        assert!(same(&p("  3.5 "), &constant(3.5)));
        assert!(same(&p(".25"), &constant(0.25)));
        assert!(same(&p("1e-3"), &constant(0.001)));
        assert!(same(&p("x"), &input(0)));
        assert!(same(&p("y"), &input(1)));
        assert!(same(&p("z"), &input(2)));
    }

    #[test]
    fn precedence_and_associativity() {
        assert!(same(&p("1 + 2 * 3"), &p("1 + (2 * 3)")));
        assert!(same(&p("1 * 2 + 3"), &p("(1 * 2) + 3")));
        assert!(same(&p("1 - 2 - 3"), &p("(1 - 2) - 3"))); // left associative
        assert!(same(&p("2 ^ 3 ^ 2"), &p("2 ^ (3 ^ 2)"))); // right associative
        assert!(same(&p("-x^2"), &p("-(x^2)")));
        assert!(same(&p("2 ^ -3"), &p("2 ^ (-3)")));
        assert!(same(&p("--x"), &p("-(-x)")));
        assert!(same(&p("+x"), &input(0)));
        assert!(same(&p("((x))"), &input(0)));
    }

    #[test]
    fn function_calls_have_any_arity() {
        assert_eq!(p("sin(x)").op, Op::Sin);
        assert_eq!(p("min(x, y)").op, Op::Min);
        assert_eq!(p("atan2(y, x)").op, Op::Atan2);
        assert_eq!(p("if(x, y, z)").op, Op::If);
        assert!(same(
            &p("exp(-x)"),
            &node(Op::Exp, vec![node(Op::Neg, vec![input(0)])])
        ));
    }

    #[test]
    fn x_index_fallback_and_constants() {
        assert!(same(&parse("x0", &[]).unwrap(), &input(0)));
        assert!(same(&parse("x7", &[]).unwrap(), &input(7)));
        // A declared variable wins over the `xN` fallback and named constants.
        let v = vec!["x1".to_string(), "e".to_string()];
        assert!(same(&parse("x1", &v).unwrap(), &input(0)));
        assert!(same(&parse("e", &v).unwrap(), &input(1)));
        assert!(same(&parse("pi", &[]).unwrap(), &constant(std::f64::consts::PI)));
    }

    #[test]
    fn errors_point_at_the_right_column() {
        assert_eq!(col("x + $"), 5);
        assert_eq!(col("foo(x)"), 1);
        assert_eq!(col("w"), 1);
        assert_eq!(col("sin(x, y)"), 1);
        assert_eq!(col("(x"), 3);
        assert_eq!(col("x @ y"), 3);
        assert_eq!(col(""), 1);
        assert_eq!(col("   "), 4);
        assert_eq!(col("x +"), 4);
    }

    #[test]
    fn error_messages_name_the_problem() {
        let e = parse("foo(x)", &vars()).unwrap_err();
        assert!(e.message.contains("unknown function"), "{}", e.message);
        let e = parse("w", &vars()).unwrap_err();
        assert!(e.message.contains("unknown variable"), "{}", e.message);
        let e = parse("sin(x, y)", &vars()).unwrap_err();
        assert!(e.message.contains("takes 1 argument"), "{}", e.message);
    }

    #[test]
    fn round_trips_through_to_string() {
        let corpus = [
            "x",
            "-x",
            "x + y",
            "x - y * z",
            "x / y",
            "x ^ y",
            "(x + y) * z",
            "sin(x) + cos(y)",
            "exp(-x)",
            "if(x, y, z)",
            "min(x, y)",
            "max(x, 3)",
            "atan2(y, x)",
            "2.5 * x + 3",
            "x ^ 2 ^ 3",
            "mod(x, y)",
            "sqrt(abs(x))",
            "x > 0",
            "x <= y",
            "x == 0",
            "(x > 0) and (y < 1)",
            "x > 0 or y > 0",
            "(x > 0) xor (y > 0)",
            "not(x > 0)",
            "not x",
            "between(x, -1, 1)",
            "require(x > 0)",
            "constrain(x)",
            "x * greater(x, 0)",
            "if((x > 0) and (y > 0), x, 0)",
            "complexity(x * x)",
            "variables(x * y)",
            "constants(x + 1)",
            "contains(sin(x))",
            "contains_op(x)",
            "complexity(complexity(x))",
        ];
        let v = vars();
        for s in corpus {
            let tree = parse(s, &v).expect(s);
            let printed = crate::expr::to_string(&tree, &v);
            let again = parse(&printed, &v)
                .unwrap_or_else(|e| panic!("re-parsing '{printed}' (from '{s}'): {e}"));
            assert!(
                same(&tree, &again),
                "'{s}' printed as '{printed}' did not round-trip"
            );
        }
    }

    #[test]
    fn conditional_precedence_and_associativity() {
        // Loosest to tightest: or < xor < and < comparisons < sum < product <
        // unary < power.
        assert!(same(&p("x > 0 and y > 0 or z > 0"), &p("((x > 0) and (y > 0)) or (z > 0)")));
        assert!(same(&p("x > 0 or y > 0 and z > 0"), &p("(x > 0) or ((y > 0) and (z > 0))")));
        assert!(same(&p("x > 0 xor y > 0 and z > 0"), &p("(x > 0) xor ((y > 0) and (z > 0))")));
        assert!(same(&p("x and y and z"), &p("(x and y) and z"))); // left associative
        assert!(same(&p("x + 1 > y * 2"), &p("(x + 1) > (y * 2)")));
        assert!(same(&p("-x > 0"), &p("(-x) > 0")));
        // `not` binds at the unary level, so it clings to one operand.
        assert!(same(&p("not x > 0"), &p("(not x) > 0")));
        assert!(same(&p("not (x > 0)"), &p("not(x > 0)")));
        // `>=` / `<=` / `==` produce the right operators.
        assert_eq!(p("x >= 0").op, Op::GreaterOrEqual);
        assert_eq!(p("x <= 0").op, Op::LessOrEqual);
        assert_eq!(p("x == 0").op, Op::Equal);
        assert_eq!(p("x and y").op, Op::And);
        assert_eq!(p("x or y").op, Op::Or);
        assert_eq!(p("x xor y").op, Op::Xor);
        assert_eq!(p("between(x, y, z)").op, Op::Between);
        assert_eq!(p("require(x)").op, Op::Require);
        assert_eq!(p("constrain(x)").op, Op::Constrain);
        // Keyword matching is case-insensitive, like function names.
        assert_eq!(p("x AND y").op, Op::And);
        assert_eq!(p("NOT x").op, Op::Not);
    }

    #[test]
    fn comparisons_are_non_associative() {
        assert!(parse("x < y < z", &vars()).is_err());
        assert!(parse("x > y > z", &vars()).is_err());
        assert!(parse("x == y == z", &vars()).is_err());
        // A parenthesised chain is fine.
        assert!(parse("(x < y) and (y < z)", &vars()).is_ok());
        // A single '=' is not an operator.
        let e = parse("x = y", &vars()).unwrap_err();
        assert!(e.message.contains("'=='"), "{}", e.message);
    }
}
