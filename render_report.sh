#!/bin/bash
# Script to render the Quarto report to PDF

echo "Rendering Quarto report to PDF..."
quarto render OIL_FORECASTING_REPORT.qmd --to pdf

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ PDF successfully created!"
    echo "Output: OIL_FORECASTING_REPORT.pdf"
else
    echo ""
    echo "✗ Error rendering PDF. Check if LaTeX is installed."
    echo ""
    echo "If LaTeX is not installed, you can:"
    echo "1. Install BasicTeX (smaller): brew install --cask basictex"
    echo "2. Or install full MacTeX (larger): brew install --cask mactex"
    echo ""
    echo "After installing, you may need to add to PATH:"
    echo "export PATH=\"/Library/TeX/texbin:\$PATH\""
fi

