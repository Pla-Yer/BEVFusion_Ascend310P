# BEVFusion-Ascend310P Documentation Guide

This directory contains all documentation for the BEVFusion-Ascend310P deployment project. This guide helps AI agents understand how to properly organize and record human-written reports.

## Directory Structure

```
docs/
├── 01_Project_Overview.md          # Project overview and objectives
├── 02_System_Architecture.md       # System architecture design
├── 03_Module_Status.md             # Module development status
├── 04_Decision_Log/                # Technical decision records
├── 05_Evaluation_Report/           # Model evaluation reports
├── 06_Performance_Report/          # Performance analysis reports
├── 07_Risk_Register/               # Risk management records
├── 08_Weekly_Report/               # Weekly progress reports
├── 09_Roadmap.md                   # Project roadmap
└── 10_Debug_Knowledge_Base/        # Debug knowledge base
```

## Document Update Rules

### Document Categories

The documentation is divided into two categories:

**Source Documents (源文档)** - Records of specific events and data:
- `04_Decision_Log/` - Technical decision records
- `05_Evaluation_Report/` - Model evaluation reports
- `06_Performance_Report/` - Performance analysis reports
- `07_Risk_Register/` - Risk management records
- `10_Debug_Knowledge_Base/` - Debug knowledge base

**Summary Documents (汇总文档)** - Synthesized from source documents:
- `01_Project_Overview.md` - Project overview and objectives
- `02_System_Architecture.md` - System architecture design
- `03_Module_Status.md` - Module development status
- `08_Weekly_Report/` - Weekly progress reports

### Update Workflow

**When instructed to "update the entire documentation" (更新整个文档), follow this workflow:**

1. **First**: Update source documents (04, 05, 06, 07, 10) based on new information
   - Create new records for new events
   - Update existing records if information changes
   - Follow the minimum scope principle

2. **Then**: Update summary documents (01, 02, 03, 08) based on source documents
   - Synthesize information from source documents
   - Update project status and progress
   - Maintain consistency with source documents

**Example**:
```
Human: "Update the entire documentation with today's work"

AI Agent should:
1. Create new records in source documents:
   - Add ERR-XXX in 10_Debug_Knowledge_Base/
   - Add EVAL-XXX in 05_Evaluation_Report/
   - etc.

2. Update summary documents:
   - Update 03_Module_Status.md with new module status
   - Update 08_Weekly_Report/ with weekly progress
   - Update 01_Project_Overview.md if needed
   - etc.
```

### Why This Order?

- **Source documents are the source of truth**: They contain detailed, specific information
- **Summary documents are derived**: They synthesize and summarize information from source documents
- **Avoid circular dependencies**: Always update source → summary, never the reverse
- **Maintain consistency**: Summary documents should always reflect the current state of source documents

## How to Record Reports

### General Principles

1. **One Record Per File**: Each decision, evaluation, performance test, risk, or weekly report should be in its own file
2. **Use Templates**: Follow the templates provided in each directory's `Readme.md`
3. **Update Index**: After creating a new record, update the `Index.md` file in that directory
4. **Use Consistent Naming**: Follow the naming convention `ID-title.md` (e.g., `D-001-replace-sparseconv3d-with-pillar.md`)
5. **Include All Required Fields**: Ensure all template fields are filled

### ⚠️ CRITICAL: Minimum Scope Principle (最小范围原则)

**This is the most important principle to follow:**

**Each piece of content should appear in ONLY ONE document.**

#### Why This Matters
- **Avoid Redundancy**: Duplicating content across multiple documents creates maintenance burden
- **Prevent Inconsistency**: When the same content exists in multiple places, updates may be missed
- **Reduce Confusion**: Readers should know exactly where to find information
- **Save Storage**: Unnecessary duplication wastes space

#### How to Follow This Principle

1. **Choose the Right Location**: Before writing, determine the most appropriate single location for the content
   - Example: A decision about model architecture goes in `04_Decision_Log/`, NOT in both Decision Log and Weekly Report

2. **Use References Instead of Duplication**: If you need to mention content that exists elsewhere, use a reference
   ```markdown
   # Good: Reference with link
   See decision D-001 for details on SparseConv3d replacement.

   # Bad: Duplicating the entire decision content
   We decided to replace SparseConv3d with Pillar because...
   [Full decision details repeated here]
   ```

3. **Cross-Reference Format**: When referencing other documents, use this format:
   ```markdown
   - For decisions: See [D-XXX](./04_Decision_Log/D-XXX-title.md)
   - For evaluations: See [EVAL-XXX](./05_Evaluation_Report/EVAL-XXX-title.md)
   - For performance: See [PERF-XXX](./06_Performance_Report/PERF-XXX-title.md)
   - For risks: See [R-XXX](./07_Risk_Register/R-XXX-title.md)
   - For debug: See [ERR-XXX](./10_Debug_Knowledge_Base/ERR-XXX-title.md)
   ```

4. **What NOT to Do**:
   ```markdown
   # ❌ WRONG: Same content in multiple places

   # In 04_Decision_Log/D-001.md:
   We chose PointPillars because it's mature and easy to implement...

   # In 08_Weekly_Report/WEEK-002.md:
   We chose PointPillars because it's mature and easy to implement...

   # In 05_Evaluation_Report/EVAL-002.md:
   We chose PointPillars because it's mature and easy to implement...
   ```

5. **What TO Do**:
   ```markdown
   # ✅ CORRECT: Content in one place, references elsewhere

   # In 04_Decision_Log/D-001.md:
   We chose PointPillars because it's mature and easy to implement...
   [Full decision details]

   # In 08_Weekly_Report/WEEK-002.md:
   Made key decision on point cloud processing (see D-001).

   # In 05_Evaluation_Report/EVAL-002.md:
   Using Pillar encoder as decided in D-001.
   ```

#### Special Cases

- **Weekly Reports**: Can summarize what was done, but should reference detailed records
  ```markdown
  # Good
  - Completed model evaluation (see EVAL-003 for details)
  - Made decision on optimization strategy (see D-004)

  # Bad
  - Completed model evaluation: mAP=0.21, NDS=0.24... [all details repeated]
  ```

- **Related Content**: If two pieces of content are related but different, they can each have their own file
  ```markdown
  # OK: Different aspects
  D-001: Decision to use Pillar (why we chose it)
  EVAL-002: Evaluation of Pillar version (how it performed)
  PERF-002: Performance of Pillar encoder (how fast it runs)
  ```

- **Updates**: When information changes, update the original document, don't create a duplicate
  ```markdown
  # Good: Update the original
  Edit D-001 to add new outcome data

  # Bad: Create new document with same content
  Create D-005 with same decision but updated outcome
  ```

### Step-by-Step Process

When a human provides a report, follow these steps:

#### Step 1: Identify Report Type

Determine which category the report belongs to:
- **Decision Log**: Technical choices, architecture decisions, algorithm selections
- **Evaluation Report**: Model accuracy, precision metrics, comparison results
- **Performance Report**: Latency analysis, optimization results, profiling data
- **Risk Register**: Project risks, mitigation strategies, contingency plans
- **Weekly Report**: Weekly progress, completed tasks, blockers
- **Debug Knowledge Base**: Error cases, debugging processes, solutions

#### Step 2: Read the Template

Navigate to the appropriate directory and read:
1. `Readme.md` - Contains the template and guidelines
2. `Index.md` - Check existing records to determine the next ID number

#### Step 3: Create the Record File

Create a new file with the naming convention:
- Format: `[ID]-[descriptive-title].md`
- Example: `D-004-model-quantization-strategy.md`
- Use lowercase and hyphens for titles
- Keep titles concise but descriptive

#### Step 4: Fill the Template

Copy the template from `Readme.md` and fill in all required fields:
- Replace placeholder text with actual content
- Use proper markdown formatting
- Include specific data, metrics, and dates
- Add relevant links and references

#### Step 5: Update the Index

Add the new record to `Index.md`:
- Add a new row in the table
- Include: ID, Title, Date, Status, File link
- Maintain chronological order (newest first or last, be consistent)

#### Step 6: Verify Completeness

Check that:
- [ ] All template fields are filled
- [ ] File naming follows convention
- [ ] Index is updated
- [ ] Links in index are correct
- [ ] Date format is consistent (YYYY-MM-DD)
- [ ] Status is appropriate (✅ Completed, 🔄 In Progress, ⏸️ Pending, ⚠️ Warning)

## Report Type Guidelines

### 04_Decision_Log

**When to use**: When documenting a technical decision that affects the project

**Required information**:
- Background and problem context
- Options considered with pros/cons
- Final choice and reasoning
- Risk assessment
- Outcome (if available)

**Example scenarios**:
- Choosing between different algorithms
- Selecting hardware configuration
- Deciding on implementation approach
- Architecture design choices

### 05_Evaluation_Report

**When to use**: When reporting model evaluation results

**Required information**:
- Test configuration (model, hardware, dataset)
- Evaluation metrics (mAP, NDS, etc.)
- Per-class results
- Comparison with baseline
- Analysis and insights

**Example scenarios**:
- Baseline evaluation
- After model optimization
- Precision comparison
- Ablation study results

### 06_Performance_Report

**When to use**: When reporting performance analysis and optimization

**Required information**:
- Test configuration
- Latency breakdown
- Bottleneck analysis
- Optimization strategies
- Before/after comparison

**Example scenarios**:
- Initial profiling
- After CPU optimization
- After NPU optimization
- Data transfer optimization

### 07_Risk_Register

**When to use**: When identifying or updating project risks

**Required information**:
- Risk level (High/Medium/Low)
- Impact description
- Probability assessment
- Mitigation strategy
- Contingency plan
- Current status

**Example scenarios**:
- New risk identified
- Risk status update
- Risk mitigation progress
- Risk resolved

### 08_Weekly_Report

**When to use**: When recording weekly progress

**Required information**:
- Week number and date range
- Goals for the week
- Completed tasks
- Blockers encountered
- Next week plan
- Progress metrics

**Example scenarios**:
- End of week summary
- Milestone completion
- Project status update

### 10_Debug_Knowledge_Base

**When to use**: When documenting error cases and solutions

**Required information**:
- Error symptoms
- Debug process
- Root cause
- Fix/solution
- Verification method
- Lessons learned

**Example scenarios**:
- Compilation errors
- Runtime errors
- Performance issues
- Precision mismatch

## Common Mistakes to Avoid

1. **❌ DON'T duplicate content across documents**: This is the WORST mistake. Each piece of information should exist in ONE place only. Use references instead of copying.
2. **Don't skip the Index**: Always update `Index.md` after creating a new record
3. **Don't use inconsistent naming**: Follow the ID-title.md format strictly
4. **Don't leave template placeholders**: Replace all `[text]` placeholders
5. **Don't forget dates**: Always include the date in YYYY-MM-DD format
6. **Don't mix report types**: Put each report in the correct directory
7. **Don't create duplicate IDs**: Check Index.md for existing IDs before creating
8. **Don't ignore status updates**: Update status when circumstances change
9. **Don't copy-paste entire sections**: If you need to reference something, use a link, not a copy

## Quick Reference

### ID Prefixes
- `D-XXX`: Decision Log
- `EVAL-XXX`: Evaluation Report
- `PERF-XXX`: Performance Report
- `R-XXX`: Risk Register
- `WEEK-XXX`: Weekly Report
- `ERR-XXX`: Debug Knowledge Base

### Status Icons
- ✅ Completed/Adopted/Resolved
- 🔄 In Progress/Mitigating
- ⏸️ Pending/Monitoring
- ⚠️ Warning/Issue
- ❌ Failed/Rejected

### Date Format
Always use: `YYYY-MM-DD` (e.g., 2026-03-07)

## Example Workflow

**Scenario**: Human says "We tested the model on 310P and got mAP=0.21"

**AI Agent should**:
1. Identify this as an Evaluation Report
2. Read `docs/05_Evaluation_Report/Readme.md` for template
3. Check `docs/05_Evaluation_Report/Index.md` for next ID (EVAL-004)
4. Create `docs/05_Evaluation_Report/EVAL-004-310p-model-test.md`
5. Fill template with:
   - Date: current date
   - Configuration: 310P hardware details
   - Metrics: mAP=0.21, other metrics if available
   - Analysis: comparison with baseline
6. Update `Index.md` with new entry
7. Confirm completion to human

## Need Help?

If unsure about how to record a report:
1. Read the `Readme.md` in the relevant directory
2. Check existing records for examples
3. Follow the template structure
4. Ask the human for clarification if needed

Remember: Good documentation is crucial for project success. Take time to record reports properly and completely.
