# Crop Disease Detection Project — Setup Guide

Follow these steps to clone and run the project on a new Windows machine using VS Code.

## 1. Install the required software

Install:
- Python (use a version compatible with the project's dependencies)
- Git
- Visual Studio Code
- The Python extension in VS Code

Verify installation in a terminal:

```bash
python --version
git --version
```

If `python` is not recognized on Windows, try `py --version`.

## 2. Clone the GitHub repository

Open a terminal and run:

```bash
git clone https://github.com/kumaditya/Crop-Disease-Detection-Service.git
cd Crop-Disease-Detection-Service
code .
```

If the repository is private, the owner must grant you access.

## 3. Create and activate a virtual environment

From the project root directory:

```bash
python -m venv .venv
```

For Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

For PowerShell or Git Bash, activation commands may differ. In PowerShell, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4. Install Python dependencies

Make sure `requirements.txt` exists in the project root, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Configure the Gemini API key

Create a `.env` file in the project root. Add your own API key and the configured model:

```env
GEMINI_API_KEY=your_own_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Get a key from Google AI Studio:
https://aistudio.google.com/apikey

Do not commit or share your `.env` file or API key. The application must load environment variables from `.env` (for example, using `python-dotenv`) and use the same variable names.

## 6. Check model files

The project's `.gitignore` contains:

```gitignore
models/*.pth
```

This means `.pth` files directly inside the `models` folder are excluded from Git tracking and may not be present after cloning. Check the `models` folder for the model files required by the application. If they are missing, ask the project owner for a trusted download/source and place them at the paths expected by the code.

Also check whether any class-label JSON or configuration files are required alongside the model.

## 7. Check dataset files if needed

The `data/` folder is not ignored by the provided `.gitignore`, so files in it may be included in the repository unless another rule excludes them. If the project requires a dataset that is not present, inspect `download_dataset.py` and follow the project's dataset download instructions.

## 8. Select the Python interpreter in VS Code

1. Press `Ctrl + Shift + P`.
2. Search for `Python: Select Interpreter`.
3. Select the interpreter inside the project's `.venv` folder.
4. Open a new terminal and ensure the virtual environment is activated.

## 9. Run the application

The repository contains `app.py`. If it is the Streamlit entry point, run:

```bash
streamlit run app.py
```

If the command is not found, ensure the virtual environment is active and dependencies were installed. If `app.py` is not a Streamlit app, check `README.md` for the correct run command.

## 10. Test the application

- Open the local URL shown in the terminal.
- Upload a crop leaf image.
- Confirm that disease prediction works.
- Confirm that the Gemini-generated explanation works, if enabled.

## Troubleshooting

- `ModuleNotFoundError`: activate `.venv` and reinstall from `requirements.txt`.
- Model/file not found: verify model files and paths in `models/`.
- Gemini authentication error: check the `.env` values and environment-variable loading.
- Python/package compatibility error: use a Python version compatible with `requirements.txt` and the project's code.

## Before sharing the project

- Keep API keys out of GitHub.
- Ensure `requirements.txt` contains the needed dependencies.
- Provide instructions or a trusted download location for ignored model weights.
- Test the steps from a fresh clone.
