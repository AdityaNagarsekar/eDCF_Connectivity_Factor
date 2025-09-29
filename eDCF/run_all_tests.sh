#!/bin/bash
# Backup the original Driver.py so we can restore it later.
cp Driver.py Driver.py.bak

# Function to set control variables in Driver.py.
set_controls() {
    local phase=$1  # "generate" or "process"
    if [ "$phase" = "generate" ]; then
        export GEN_VAL="True"
        export PROC_VAL="False"
    else
        export GEN_VAL="False"
        export PROC_VAL="True"
    fi

    python3 - <<'EOF'
import os, re

with open("Driver.py", "r") as f:
    content = f.read()

def update_flag(content, var, value):
    pattern = r"^(\s*self\." + var + r"\s*(?::\s*[^=]+)?=\s*).*$"
    return re.sub(pattern, r"\1" + value, content, flags=re.MULTILINE)

# For generate phase, update these two.
for var in ["generate_data_ctrl", "linear_transform_ctrl"]:
    content = update_flag(content, var, os.environ.get("GEN_VAL", "False"))
# For all other control variables, update them to PROC_VAL.
controls = ["algorithm_train_ctrl", "calculate_grid_params_ctrl", "compute_grid_ctrl",
            "extract_boundary_ctrl", "detect_fractal_ctrl", "calc_connectivity_ctrl",
            "data_display_ctrl", "grid_display_ctrl", "display_ctrl", "force_grid_ctrl",
            "direct_conversion_ctrl", "range_analysis_ctrl", "range_display_ctrl",
            "display_object_grid_ctrl", "force_grid_connectivity_ctrl", "topological_dimension_ctrl",
            "fractal_object_ctrl", "interpret_ctrl", "save_force_ctrl", "save_ctrl"]
for var in controls:
    content = update_flag(content, var, os.environ.get("PROC_VAL", "False"))

with open("Driver.py", "w") as f:
    f.write(content)
EOF
}

# Function to update Driver.py using a template file.
# This replaces the block from the line starting with "self.data_objects" to the line starting with "self.directory_name"
# with the reindented contents of the template file.
update_from_template() {
    local template_file="$1"
    export TEMPLATE_FILE="$template_file"
    python3 - <<'EOF'
import os, re

template_file = os.environ.get("TEMPLATE_FILE")
with open(template_file, "r") as f:
    template_lines = f.readlines()

with open("Driver.py", "r") as f:
    content = f.read()

# Pattern to match the block from self.data_objects to self.directory_name (inclusive).
pattern = r"(?sm)^(?P<indent>\s*)self\.data_objects.*?^\s*self\.directory_name.*?$"
match = re.search(pattern, content, flags=re.MULTILINE|re.DOTALL)
if match:
    indent = match.group("indent")
    # Reindent each line of the template to use the captured indent.
    new_block = "".join(indent + line.lstrip() for line in template_lines)
    content = re.sub(pattern, new_block, content, flags=re.MULTILINE|re.DOTALL)
with open("Driver.py", "w") as f:
    f.write(content)
EOF
}

# Function to process a single template file.
# For the first template in a set (algo_num == 1), it will run both the generate and process phases.
# For subsequent template files in the same set, only the process phase will be run.
process_template() {
    local template_file="$1"
    local set_num="$2"
    local algo_num="$3"

    echo "Processing Set $set_num, Algorithm $algo_num: $(basename "$template_file")"

    # Update Driver.py with the block from the template.
    update_from_template "$template_file"

    if [ "$algo_num" -eq 1 ]; then
        echo "  Phase 1: Generating data..."
        set_controls "generate"
        python3 Driver.py
    fi

    echo "  Phase 2: Processing data..."
    set_controls "process"
    python3 Driver.py

    echo "  Completed Set $set_num, Algorithm $algo_num"
    echo "----------------------------------------"
}

# Main loop to process all sets.
for set_num in {1..16}; do
    echo "========================================"
    echo "Starting Set $set_num"
    echo "========================================"
    if [ ! -d "TestingTemplates/Set$set_num" ]; then
        echo "Warning: TestingTemplates/Set$set_num not found. Skipping."
        continue
    fi
    # Get a sorted list of .txt template files.
    template_files=( $(find "TestingTemplates/Set$set_num" -name "*.txt" | sort) )
    for ((algo_num=1; algo_num<=${#template_files[@]}; algo_num++)); do
        template_file="${template_files[$((algo_num-1))]}"
        if [ -f "$template_file" ]; then
            process_template "$template_file" "$set_num" "$algo_num"
        fi
    done
    echo "Set $set_num processed successfully"
done

# Restore the original Driver.py.
mv Driver.py.bak Driver.py
echo "All 16 datasets with 4 algorithms each have been processed successfully!"