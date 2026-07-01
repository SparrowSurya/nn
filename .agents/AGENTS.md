# Project Rules & Customizations

This file contains style guidelines, structural constraints, and behavioral instructions for AI agents working on this workspace.

---

## 1. File & Edit Permissions
* **Explicit Editing Approvals**: Do **NOT** edit any file unless the user has explicitly instructed you to modify it in the current prompt. If you believe secondary files need to be edited to complete a task, explain why and ask for explicit user permission first.
* **Respect User Modifications**: Always respect and preserve any changes made by the user to code, structures, or documentation after an agent has written them. Do not overwrite or revert user refinements.

---

## 2. Execution & Command Constraints
* **No Script Execution**: Do **NOT** run the main python script (`main.py` or task programs) under any circumstances. Never run the main program.
* **Terminal Command Approvals**: Always propose terminal commands and ask for explicit user permission before executing any command or script.

---

## 3. Quality & Type Safety
* **Always Run Pyright**: Always run static type-checking (`uv run pyright`) after making any code changes to guarantee the codebase remains 100% type-safe with `0 errors`.
* **Python Version & Generics**: The project uses Python 3.12+ / 3.14. Always use PEP 695 Type Parameter syntax for generic classes (using square brackets, e.g. `class NeuralNetworkProgram[T_In, T_Out](ABC)`) instead of old `typing.TypeVar`/`typing.Generic` imports.

---

## 4. Documentation Constraints
* **Strict Conceptual Separation**: All study documentation, notes, and learning checklists placed inside the `doc/` folder (such as `doc/NOTES.md`, `doc/STUDY_PLAN.md`) must remain strictly conceptual and mathematical.
* Do **NOT** include Python code blocks, file imports, or direct references to Python class/method names inside the `doc/` folder. All code-level documentation belongs in inline comments and module docstrings.

---

## 5. Directory & Package Architecture
* **Programs Directory**: The `programs/` directory is used strictly to store independent task scripts (like `programs/xor.py` and `programs/digit_recogniser.py`). Do **NOT** create an `__init__.py` file inside `programs/` or treat it as a python package module.
* **Lib Package**: All shared neural network classes (layers, network, activations, losses, observers, visualizations, program blueprint) belong inside the `lib/` package.

---

## 6. Interaction & Communication
* **No Direct Solutions for Hints**: When the user asks for hints or learning assistance, do **NOT** provide direct copy-paste code solutions. Explain the underlying mathematical or architectural concepts and guide the user to implement the solution themselves.
* **Clickable File Links**: Always use the `file://` scheme to reference files (e.g., `[main.py](file:///Users/user/lab/nn/main.py)`). Do not wrap the link text in backticks.
* **Dynamic Rule Learning**: Always update this `AGENTS.md` file immediately whenever the user asserts that you should "do this thing from now on" or establishes new workspace-wide constraints.
* **Always Present Implementation Plan**: Before implementing any code changes or file modifications, you MUST present a step-by-step implementation plan in the chat for the user to review, whether or not the user explicitly asks for it.
