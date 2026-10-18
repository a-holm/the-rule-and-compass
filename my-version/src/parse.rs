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
//! expr    := term (('+' | '-') term)*
//! term    := unary (('*' | '/') unary)*
//! unary   := ('-' | '+') unary | power
//! power   := primary ('^' unary)?          // right associative
//! primary := number
//!          | name '(' expr (',' expr)* ')' // function call, any arity
//!          | name                          // variable
//!          | '(' expr ')'
//! ```
//!
//! Operator precedence, tightest first: `^`, unary `-`/`+`, `*`/`/`, `+`/`-`.
//! So `-x^2` is `-(x^2)` and `2^3^2` is `2^(3^2)`.
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
            _ => self.power(),
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
}
