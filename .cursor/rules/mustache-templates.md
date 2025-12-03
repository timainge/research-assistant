# Mustache Template Syntax

When using Chevron (Python Mustache implementation) for prompt templates:

## Conditionals

Use section syntax, not `if` syntax:

```mustache
{{#context}}
This shows if context is truthy
{{/context}}
```

**NOT**:
```mustache
{{#if context}}
This will ERROR
{{/if}}
```

## Loops

Same syntax for iterating arrays:

```mustache
{{#items}}
- {{name}}: {{value}}
{{/items}}
```

## Inverted Sections (else/unless)

Use `^` for inverted:

```mustache
{{#results}}
Found results
{{/results}}
{{^results}}
No results found
{{/results}}
```

## Variable Access in Loops

Inside a loop, access properties directly:

```mustache
{{#tasks}}
Task: {{id}} - {{query}}
{{/tasks}}
```

## Reference

- Chevron follows Mustache spec: https://mustache.github.io/mustache.5.html
- No helpers, partials, or advanced features like Handlebars

