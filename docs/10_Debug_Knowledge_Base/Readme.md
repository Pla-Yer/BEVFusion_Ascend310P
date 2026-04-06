# Debug Knowledge Base

This directory records important debugging cases encountered during the BEVFusion-Ascend310P deployment project.

The goal is to document:

- Environment issues
- Deployment pipeline bugs
- Operator implementation problems
- Precision mismatch issues
- Performance bottlenecks

Each error record contains:

- Symptom
- Debugging methodology
- Root cause
- Fix
- Lessons learned

These records serve as an internal engineering reference and debugging knowledge base.

For error reporting, please use the following template and add an index to Index.md:
## Report Template


```markdown
## ERR-XXX: [Error Name]

### Date
YYYY-MM-DD

### Environment
- OS:
- Version:
- Component:

### Symptom
[Error message, log, behavior, impact]

### Debug Process(if created scripts, please attach them here) 
1.
2.
3.

### Root Cause
[Final confirmed root cause]

### Fix
1.
2.

### Verification
[How to confirm it’s fixed]

### Lessons & Notes
[Key takeaway]
```