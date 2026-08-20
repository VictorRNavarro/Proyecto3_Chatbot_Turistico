"""Ejecuta un notebook y conserva sus salidas para la entrega."""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python tests/ejecutar_notebook.py notebooks/archivo.ipynb")

    ruta = Path(sys.argv[1])
    notebook = nbformat.read(ruta, as_version=4)
    cliente = NotebookClient(notebook, timeout=1800, kernel_name="python3", allow_errors=False)
    cliente.execute(cwd=str(Path.cwd()))
    nbformat.write(notebook, ruta)
    print(f"EJECUTADO: {ruta}")


if __name__ == "__main__":
    main()
