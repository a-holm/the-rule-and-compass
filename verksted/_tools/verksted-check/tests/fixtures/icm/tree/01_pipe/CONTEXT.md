---
type: pipeline # trailing comment dropped
title: "Pipe \"one\" å"
description: 'Single quoted: kept as is'
tags: [alpha, beta]
consumes: [../_shared/rules.md, ../_shared/gone.md]
produces: work/result.md
---
# Pipe title

One job.

## Inputs

- Reference: `../_shared/rules.md` and `_shared/rules.md`
- reference list `../_shared/nope.md`
- per-run: `work/run.md` (folder exists)
- per-run root-relative: `01_pipe/work/other.md`
- per-run without folder: `nofolder/run.md`
* star bullet, reference: `star-gone.md`
* star bullet per-run: `work/star.md`
- not paths: `plain` `{holder}/x.md` `https://x.y/z.md` `a|b/c` `*.md` `file:x.md` `<tag>/y.md`
- extension only: `notes.txt` and `v1.2.3`
- dir path per-run: `work/`
Text line with `not/a/bullet.md` is not checked.

## Process

1. [Stage a](stages/01_a/CONTEXT.md)
2. [Stage b](stages/02_b/CONTEXT.md) and again [stage b](stages/02_b/CONTEXT.md)

## Inputs ##

- reference `second/inputs-gone.md`

## Notes

- Not an inputs section: `notes-gone.md`
