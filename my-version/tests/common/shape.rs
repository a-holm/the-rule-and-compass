//! Report comparison that ignores measured time and number print forms.

use rcsr::devlog::Json;

fn is_timing_key(key: &str) -> bool {
    matches!(key, "ledger" | "timing" | "elapsed" | "seconds") || key.starts_with("elapsed_") || key.ends_with("_s") || key.ends_with("_seconds")
}

fn render(j: &Json, out: &mut String) {
    match j {
        Json::Num(_) => out.push('#'),
        Json::Arr(items) => {
            out.push('[');
            for item in items {
                render(item, out);
                out.push(',');
            }
            out.push(']');
        }
        Json::Obj(members) => {
            out.push('{');
            for (key, value) in members.iter().filter(|(k, _)| !is_timing_key(k)) {
                out.push_str(&format!("{key:?}:"));
                render(value, out);
                out.push(',');
            }
            out.push('}');
        }
        other => out.push_str(&format!("{other:?}")),
    }
}

/// Structure, keys, strings and booleans of `j`; every number is masked and
/// every timing member (see `is_timing_key`) is dropped at any depth.
pub fn timing_free_shape(j: &Json) -> String {
    let mut out = String::new();
    render(j, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Json {
        Json::parse(text).expect("json")
    }

    #[test]
    fn timing_print_form_does_not_matter() {
        let a = parse(r#"{"rungs":[{"name":"r","wall_seconds":3.2e-5,"fit":{"host_s":0.5}}],"ledger":{"x":1}}"#);
        let b = parse(r#"{"rungs":[{"name":"r","wall_seconds":0.0123,"fit":{"host_s":3}}]}"#);
        assert_eq!(timing_free_shape(&a), timing_free_shape(&b));
    }

    #[test]
    fn digit_strip_text_differs_on_print_form() {
        let strip = |j: &Json| format!("{j:?}").chars().filter(|c| !c.is_ascii_digit()).collect::<String>();
        let a = parse(r#"{"wall_seconds":3.2e-5}"#);
        let b = parse(r#"{"wall_seconds":0.0123}"#);
        assert_ne!(strip(&a), strip(&b));
    }

    #[test]
    fn data_difference_still_differs() {
        let a = parse(r#"{"rungs":[{"name":"r","wall_seconds":1.0,"policy":"a"}]}"#);
        let b = parse(r#"{"rungs":[{"name":"r","wall_seconds":1.0,"policy":"b"}]}"#);
        let c = parse(r#"{"rungs":[{"name":"r","wall_seconds":1.0}]}"#);
        assert_ne!(timing_free_shape(&a), timing_free_shape(&b));
        assert_ne!(timing_free_shape(&a), timing_free_shape(&c));
    }
}
