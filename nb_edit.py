"""
nb_edit.py — Fine-grained Jupyter notebook cell editor and runner.

Usage:
    python nb_edit.py <notebook> list
    python nb_edit.py <notebook> read <cell_number>
    python nb_edit.py <notebook> replace <cell_number> <code_file>
    python nb_edit.py <notebook> insert <cell_number> <code_file> [--type markdown]
    python nb_edit.py <notebook> delete <cell_number>
    python nb_edit.py <notebook> run <cell_number> [--timeout 120]
    python nb_edit.py <notebook> run-range <start> <end> [--timeout 120]
    python nb_edit.py <notebook> output <cell_number>

Cell numbers are 0-indexed.

Examples:
    python nb_edit.py Phase2B_Models.ipynb list
    python nb_edit.py Phase2B_Models.ipynb read 5
    python nb_edit.py Phase2B_Models.ipynb output 5
    python nb_edit.py Phase2B_Models.ipynb replace 5 fix.py
    python nb_edit.py Phase2B_Models.ipynb run 5
    python nb_edit.py Phase2B_Models.ipynb run-range 3 7
"""

import argparse
import json
import sys
import subprocess
import tempfile
import os


def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def get_source(cell):
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def set_source(cell, text):
    cell["source"] = text.splitlines(True)


def get_outputs_text(cell):
    """Extract readable text from cell outputs."""
    lines = []
    for out in cell.get("outputs", []):
        otype = out.get("output_type", "")
        if otype == "stream":
            lines.append(f"[{out.get('name', 'stdout')}]")
            text = out.get("text", [])
            if isinstance(text, list):
                lines.append("".join(text))
            else:
                lines.append(text)
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                tp = data["text/plain"]
                if isinstance(tp, list):
                    lines.append("".join(tp))
                else:
                    lines.append(tp)
            if "image/png" in data:
                lines.append("[image/png output]")
        elif otype == "error":
            lines.append(f"[ERROR] {out.get('ename', '')}: {out.get('evalue', '')}")
            tb = out.get("traceback", [])
            for t in tb:
                lines.append(t)
    return "\n".join(lines)


def cmd_list(nb, args):
    cells = nb["cells"]
    print(f"Notebook has {len(cells)} cells\n")
    for i, cell in enumerate(cells):
        ctype = cell["cell_type"]
        src = get_source(cell)
        first_line = src.split("\n")[0][:80] if src.strip() else "(empty)"
        n_outputs = len(cell.get("outputs", []))
        has_img = any(
            "image/png" in out.get("data", {})
            for out in cell.get("outputs", [])
            if out.get("output_type") in ("display_data", "execute_result")
        )
        markers = []
        if n_outputs:
            markers.append(f"{n_outputs} outputs")
        if has_img:
            markers.append("has image")
        marker_str = f"  [{', '.join(markers)}]" if markers else ""
        print(f"  [{i:3d}] {ctype:10s} | {first_line}{marker_str}")


def cmd_read(nb, args):
    idx = args.cell
    cells = nb["cells"]
    if idx < 0 or idx >= len(cells):
        print(f"Error: cell {idx} out of range (0-{len(cells)-1})", file=sys.stderr)
        sys.exit(1)
    cell = cells[idx]
    print(f"--- Cell {idx} ({cell['cell_type']}) ---")
    print(get_source(cell))


def cmd_output(nb, args):
    idx = args.cell
    cells = nb["cells"]
    if idx < 0 or idx >= len(cells):
        print(f"Error: cell {idx} out of range (0-{len(cells)-1})", file=sys.stderr)
        sys.exit(1)
    cell = cells[idx]
    text = get_outputs_text(cell)
    if text.strip():
        print(f"--- Output of Cell {idx} ---")
        print(text)
    else:
        print(f"Cell {idx} has no text output.")


def cmd_replace(nb, args):
    idx = args.cell
    cells = nb["cells"]
    if idx < 0 or idx >= len(cells):
        print(f"Error: cell {idx} out of range (0-{len(cells)-1})", file=sys.stderr)
        sys.exit(1)
    with open(args.code_file, "r", encoding="utf-8") as f:
        new_code = f.read()
    set_source(cells[idx], new_code)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    save_notebook(nb, args.notebook)
    print(f"Replaced cell {idx} with contents of {args.code_file}")


def cmd_insert(nb, args):
    idx = args.cell
    cells = nb["cells"]
    if idx < 0 or idx > len(cells):
        print(f"Error: position {idx} out of range (0-{len(cells)})", file=sys.stderr)
        sys.exit(1)
    with open(args.code_file, "r", encoding="utf-8") as f:
        new_code = f.read()
    ctype = args.type or "code"
    new_cell = {
        "cell_type": ctype,
        "metadata": {},
        "source": new_code.splitlines(True),
    }
    if ctype == "code":
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
    cells.insert(idx, new_cell)
    save_notebook(nb, args.notebook)
    print(f"Inserted {ctype} cell at position {idx}")


def cmd_delete(nb, args):
    idx = args.cell
    cells = nb["cells"]
    if idx < 0 or idx >= len(cells):
        print(f"Error: cell {idx} out of range (0-{len(cells)-1})", file=sys.stderr)
        sys.exit(1)
    removed = cells.pop(idx)
    save_notebook(nb, args.notebook)
    print(f"Deleted cell {idx} ({removed['cell_type']})")


def run_cells(notebook_path, start, end, timeout=120):
    """Execute cells [start, end] in-place using nbconvert preprocessor."""
    nb = load_notebook(notebook_path)
    cells = nb["cells"]

    if start < 0 or end >= len(cells) or start > end:
        print(f"Error: invalid range [{start}, {end}] for {len(cells)} cells", file=sys.stderr)
        sys.exit(1)

    # Build a temporary notebook with:
    #   1) All cells before 'start' (for side effects / imports) marked as already executed
    #   2) Target cells to execute
    # We use a wrapper script approach for reliability.
    script = []
    # Collect source from all prior code cells (for context)
    for i in range(start):
        if cells[i]["cell_type"] == "code":
            script.append(get_source(cells[i]))

    # Separator
    script.append("\n# --- CONTEXT ABOVE, TARGET CELLS BELOW ---\n")

    target_sources = []
    for i in range(start, end + 1):
        if cells[i]["cell_type"] == "code":
            target_sources.append(get_source(cells[i]))

    # Create a mini notebook with all cells for execution
    tmp_nb = json.loads(json.dumps(nb))  # deep copy

    # Mark cells outside the range to skip during execution by converting to raw
    # Actually, the simplest reliable approach: use nbconvert with cell tags
    # But even simpler: run only the target cells as a script and capture output

    # Use the direct execution approach
    combined = "\n".join(target_sources)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        # Add context cells
        for i in range(start):
            if cells[i]["cell_type"] == "code":
                tf.write(get_source(cells[i]))
                tf.write("\n\n")
        tf.write("# ===== TARGET CELLS =====\n")
        tf.write(combined)
        tmp_path = tf.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(notebook_path)) or ".",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        print(f"--- Execution of cells [{start}:{end}] ---")
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"\n[stdout]\n{result.stdout}")
        if result.stderr:
            # Filter out common warnings
            stderr_lines = result.stderr.split("\n")
            important = [
                l for l in stderr_lines
                if l.strip() and not l.startswith("Traceback") or "Error" in l
            ]
            print(f"\n[stderr]\n{result.stderr}")

        # Update the cell outputs in the notebook
        stdout_output = result.stdout
        stderr_output = result.stderr

        # Store combined output in the last target cell
        last_target = end
        while last_target >= start and cells[last_target]["cell_type"] != "code":
            last_target -= 1

        if last_target >= start:
            new_outputs = []
            if stdout_output.strip():
                new_outputs.append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": stdout_output.splitlines(True),
                })
            if result.returncode != 0 and stderr_output.strip():
                new_outputs.append({
                    "output_type": "stream",
                    "name": "stderr",
                    "text": stderr_output.splitlines(True),
                })
            cells[last_target]["outputs"] = new_outputs
            save_notebook(nb, notebook_path)
            print(f"\nOutputs saved to cell {last_target}")

    except subprocess.TimeoutExpired:
        print(f"Error: execution timed out after {timeout}s", file=sys.stderr)
        sys.exit(1)
    finally:
        os.unlink(tmp_path)


def cmd_run(nb, args):
    run_cells(args.notebook, args.cell, args.cell, timeout=args.timeout)


def cmd_run_range(nb, args):
    run_cells(args.notebook, args.start, args.end, timeout=args.timeout)


def safe_print(text):
    """Print text, replacing unencodable characters."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def main():
    # Force UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Jupyter notebook cell editor/runner")
    parser.add_argument("notebook", help="Path to .ipynb file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all cells")

    p_read = sub.add_parser("read", help="Read cell source")
    p_read.add_argument("cell", type=int)

    p_out = sub.add_parser("output", help="Read cell output")
    p_out.add_argument("cell", type=int)

    p_rep = sub.add_parser("replace", help="Replace cell source from file")
    p_rep.add_argument("cell", type=int)
    p_rep.add_argument("code_file")

    p_ins = sub.add_parser("insert", help="Insert new cell at position")
    p_ins.add_argument("cell", type=int)
    p_ins.add_argument("code_file")
    p_ins.add_argument("--type", choices=["code", "markdown"], default="code")

    p_del = sub.add_parser("delete", help="Delete cell")
    p_del.add_argument("cell", type=int)

    p_run = sub.add_parser("run", help="Run a single cell")
    p_run.add_argument("cell", type=int)
    p_run.add_argument("--timeout", type=int, default=120)

    p_rr = sub.add_parser("run-range", help="Run cells in range [start, end]")
    p_rr.add_argument("start", type=int)
    p_rr.add_argument("end", type=int)
    p_rr.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()
    nb = load_notebook(args.notebook)

    commands = {
        "list": cmd_list,
        "read": cmd_read,
        "output": cmd_output,
        "replace": cmd_replace,
        "insert": cmd_insert,
        "delete": cmd_delete,
        "run": cmd_run,
        "run-range": cmd_run_range,
    }
    commands[args.command](nb, args)


if __name__ == "__main__":
    main()
