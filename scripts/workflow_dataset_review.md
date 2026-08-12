<!-- ## Workflow of the dataset review

1. Extract the consumption label data from the downloaded json file 

1.1. Use the correct name for the json file in the extract_consumption_index.py script 
1.2. Go through that now readable json file compare for a view values whether it actually worked

2. Check for issues

2.1. Run the validate_data_consumed.py script. It list how many total erros you have and list the first ten errors.
2.2. Analyse if you have issues you are done. When you have 0 issues you are done and can skrip step 3. Otherwise continue with step 3.

3. Fix issues
3.1. Run the script review_consumption_webui.py it shows all files without issues green and with issues red. TODO: One could add a toggle.
Only the red tasks are important so those with issues. 
3.2. Click on the issue that is red. Read at the top the problem. Look at the image with its labels evaluate whether the flagged problem actually exists. Below you can manipulate the json file. If the manipulation isnt possible in this UI at the top you have the image id which you can then use to scan through your  -->

# Enhanced Workflow for Dataset Review

## Overview

This comprehensive workflow guides you through the complete process of reviewing and validating consumption label data extracted from Label Studio. The dataset review process ensures data quality by systematically identifying and correcting annotation errors, inconsistencies, and missing values before analysis or model training.

The workflow consists of three main phases: extracting and verifying raw consumption data, running automated validation checks to identify issues, and then manually reviewing and fixing any flagged records through an interactive web interface. This multi-stage approach balances automation with human judgment to catch both systematic errors and context-specific anomalies that automated checks might miss.

### How Consumption Label Dataset Review Works

**Consumption labels** represent annotations for how much of a product or substance has been consumed from a container, typically shown in images of trays or packaging. Each label includes:

- **Numeric consumption values**: The actual amount consumed, measured in standardized units
- **Binary choice states**: Boolean flags indicating whether consumption occurred, whether a mask is present, or other binary annotations
- **Visual context**: The associated image of the tray showing the consumption state

The review process validates that these three components are consistent and accurate. For example, if a consumption value is zero, a corresponding "no consumption" mask should be present in the image. Similarly, mutually exclusive choices (like "consumed" and "not consumed") should never both be marked as true for the same record.

The automated validation phase scans the entire dataset to flag these types of logical inconsistencies. The manual review phase then allows human reviewers to examine flagged cases visually, determine whether the anomaly is a genuine annotation error or a valid edge case, and make corrections directly in the web interface.

---

## Phase 1: Extract and Verify Consumption Label Data

This initial phase sets up your data extraction pipeline and performs basic validation checks to ensure your source data is correctly processed into a standardized format.

### Step 1.1: Configure Export Source

Open the `extract_consumption_index.py` script in your editor. Locate the input path configuration section and verify that it accurately points to your downloaded **Label Studio JSON export file**. This file contains the raw annotations exported directly from Label Studio and serves as the authoritative source for your consumption labels.

Common configuration locations include:
- `input_path = "path/to/label_studio_export.json"`
- Environment variables or a separate config file that specifies the export location

Ensure the path is absolute or relative from the script's working directory to avoid path resolution errors during execution.

### Step 1.2: Execute Extraction

Once your input path is configured, **run the extraction script** from your terminal:

```bash
python extract_consumption_index.py
```

The script will read the raw Label Studio export, process each annotation, and generate a derived consumption index file at `data_consumed/consumption_index.json`. This JSON file represents your standardized dataset with all consumption labels organized in a consistent format for downstream validation and review.

During execution, the script may print progress indicators or summary statistics. Note any warnings or errors, as these may indicate malformed records in your source data that require upstream fixes.

### Step 1.3: Spot-Check the Index

**Open the generated `consumption_index.json` file** in a JSON viewer or text editor and manually inspect a few random records (identified by their image stems or IDs) to verify correct data extraction.

For each spot-checked record, compare:
- **Field names and structure**: Confirm that all expected fields (e.g., `consumption_value`, `has_mask`, `choice_a`, `choice_b`) are present
- **Numeric values**: Check that consumption amounts are reasonable and correctly transcribed from the original annotations
- **Choice mappings**: Verify that boolean or categorical choices have been correctly converted from Label Studio's internal representation

If you notice systematic misalignment in how fields or values have been mapped, stop here and revise the extraction script before proceeding. This spot-check acts as an early warning system to prevent cascading errors in later validation phases.

---

## Phase 2: Run Automated Validation Checks

Automated validation systematically scans your entire dataset to identify logical inconsistencies, missing values, and constraint violations that indicate potential annotation errors.

### Step 2.1: Execute the Validator

**Run the validation script** via your terminal:

```bash
python scripts/validate_data_consumed.py
```

The validation script will:
- **Parse the consumption_index.json** file you generated in Phase 1
- **Apply validation rules** to each record
- **Log all discovered issues** to `data_consumed/audit_logs/` in both human-readable and JSON formats
- **Print a summary** showing the total error count and the first 10 issues to your terminal for quick review

#### Validation Rules

The validator enforces three core rules to catch logical inconsistencies:

1. **Completeness and Mapping Rule**: Records must correctly map fields, choices, and numbers from the source data without structural anomalies or missing required data points.

2. **Zero-Consumption Mask Rule**: A zero-consumption value must not have a missing mask; missing a mask when consumption is valued at zero is flagged as an issue.

3. **Choice State Exclusivity Rule**: Choice states must not violate mutually exclusive conditions (having conflicting choice states set simultaneously is flagged as an issue).


The audit log provides traceable records of which records failed which validations, enabling you to prioritize fixes and track corrections over time.

### Step 2.2: Evaluate Audit Results

After running the validator, **review the output summary**:

- **If the total issue count is 0**: Your dataset has passed all automated checks. Validation is complete, and you can proceed directly to analysis or model training without manual review.
  
- **If issues exist** (count > 0): Proceed to Phase 3 to manually review and correct the flagged records. The number of issues will inform how much manual review work lies ahead. A few issues (<5) may be quickly fixable; a larger number (>50) may warrant investigating whether a systematic error exists in your extraction or annotation process.

---

## Phase 3: Review and Fix Issues via Web UI

The interactive web interface allows human reviewers to visually inspect flagged records, understand why they were flagged, and make corrections with full awareness of the image context.

### Step 3.1: Launch the Local Review Tool

**Start the interactive web interface** by running:

```bash
python scripts/review_consumption_webui.py
```

This command launches a lightweight local web server and opens an interactive dashboard in your default browser. The interface displays your flagged records alongside their associated tray images, allowing you to make informed decisions about whether each flag represents a genuine error or a valid edge case.

The server runs locally on your machine, ensuring all data remains private and there is no network latency when reviewing images or updating records.

### Step 3.2: Filter and Prioritize Tasks

**Open your browser** at `http://127.0.0.1:8765` (or the URL printed in your terminal if the port differs).

On the main dashboard, you will see:
- **A sidebar task list** showing all records from your dataset
- **A toggle switch** labeled **"Show only tasks with issues"**

**Enable the filter toggle** to hide all clean records and display only those flagged by the validator. This dramatically reduces cognitive load by focusing your review effort on records that need attention.

Visual indicators help you prioritize:
- **Red highlighting**: Tasks with validation issues (require review and potential correction)
- **Green highlighting**: Clean tasks that passed all validation checks (no action needed)

### Step 3.3: Inspect and Correct Flagged Records

Work through each red-flagged task using this systematic process:

1. **Click on a red task** in the sidebar to load its record details and associated image.

2. **Read the specific validation problem description** displayed prominently at the top of the review panel. This explains exactly which constraint or rule the record violated (e.g., "Consumption value is 0 but no_consumption_mask is false").

3. **Inspect the rendered tray image** carefully to visually evaluate whether the flagged anomaly is a genuine annotation error or a valid edge case:
   - If consumption is flagged as zero but the image shows product clearly remaining, the flag is likely valid
   - If a mutual exclusivity violation is flagged but the image suggests both states could reasonably apply, investigate upstream annotation practices
   - Note any visual ambiguities or unclear areas of the image that make annotation judgment difficult

4. **Correct the fields** directly in the review interface using the input forms below the image:
   - **Numbers section**: Update numeric consumption values by typing a corrected value
   - **Choices section**: Toggle boolean or categorical choices using checkboxes or dropdown selectors
   - **Delete checkbox**: Check the Delete box next to any numeric field to remove an invalid entry entirely (useful for erasing spurious values)

5. **Consider an alternative fallback** if the record is too complex to fix in the UI:
   - Note the **task ID / image stem** displayed in the record header
   - Locate this record in your source files (Label Studio project or original annotations)
   - Fix the underlying annotation or metadata upstream
   - Re-run the extraction and validation phases to propagate the upstream fix

6. **Click "Save Task Changes"** to commit your edits to the current record. The interface will validate your changes immediately and provide feedback if new issues arise.

7. **Repeat steps 1–6** for all remaining red-flagged tasks in the sidebar.

8. **Click "Save JSON"** in the top toolbar when you have finished reviewing all tasks. This persists all corrections back to `consumption_index.json`, replacing the original version with your corrected dataset.

### Step 3.4: Verify Corrections (Optional but Recommended)

After saving, you may optionally **re-run the validation script** to confirm that your corrections have resolved the flagged issues:

```bash
python scripts/validate_data_consumed.py
```

If the error count drops to zero or near-zero, you have successfully completed the dataset review. If issues persist, you may have missed some records or introduced new inconsistencies during correction—revisit the web UI to address remaining problems.

---

## Summary and Next Steps

Once you have completed all three phases:

1. Your `data_consumed/consumption_index.json` file is clean, validated, and ready for analysis
2. Your `data_consumed/audit_logs/` directory contains a permanent record of all issues identified and corrected
3. You can confidently use this dataset for training models, generating reports, or conducting further analysis

If you discover systematic issues during review (e.g., a particular annotation pattern that is consistently wrong), consider investigating whether the Label Studio project instructions or interface led annotators astray. Updating those instructions and re-annotating affected records may be more efficient than correcting individual cases one-by-one.

---