# Fixture workshop

Entry file. Good links: [pipeline](01_pipe/CONTEXT.md), [stage](<01_pipe/stages/01_a/CONTEXT.md>), [frag](01_pipe/CONTEXT.md#process), [spaced](_shared/with%20space.md), [tool](01_pipe/stages/01_a/tool.sh), [dir](01_pipe/stages), [self](AGENTS.md).

Skipped: [web](https://example.com/x.md), [anchor](#top), [mail](mailto:a@b.c), [holder]({name}.md), [wiki]([[note]]).

Broken: [gone](missing.md), [gone sh](01_pipe/stages/01_a/nothing.sh) and [again](missing.md).

Not matched by the link pattern: [titled](CONTEXT.md "a title") and [empty]().

Same target twice: [p1](01_pipe/CONTEXT.md) and [p2](01_pipe/CONTEXT.md).

Inline code is still scanned: `[x](in-code-gone.md)`.

```
[fenced](fenced-gone.md)
```

~~~
[tilde fenced](tilde-gone.md)
~~~

[after fences](after-gone.md)

See the [references shelf](references/a.md).
