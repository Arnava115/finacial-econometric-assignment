# How to Render the Report to PDF

## Quick Start

You have a Quarto document (`OIL_FORECASTING_REPORT.qmd`) ready to render. Here are your options:

## Option 1: Install LaTeX (Recommended for Best Quality)

### Install BasicTeX (Smaller, ~100MB)
```bash
brew install --cask basictex
```

After installation, add to your shell profile (`~/.zshrc` or `~/.bash_profile`):
```bash
export PATH="/Library/TeX/texbin:$PATH"
```

Then reload your shell:
```bash
source ~/.zshrc  # or source ~/.bash_profile
```

### Or Install Full MacTeX (Larger, ~4GB, but more complete)
```bash
brew install --cask mactex
```

### Then Render:
```bash
quarto render OIL_FORECASTING_REPORT.qmd --to pdf
```

Or use the script:
```bash
./render_report.sh
```

## Option 2: Use Quarto's HTML to PDF (No LaTeX Needed)

If you don't want to install LaTeX, you can render to HTML first, then convert:

```bash
# Render to HTML
quarto render OIL_FORECASTING_REPORT.qmd --to html

# Then use your browser's print-to-PDF function
# Or use a tool like wkhtmltopdf if installed
```

## Option 3: Use VS Code Quarto Extension

If you use VS Code:
1. Install the "Quarto" extension
2. Open `OIL_FORECASTING_REPORT.qmd`
3. Click "Render" button in the toolbar
4. Select PDF output

## Troubleshooting

### "xelatex not found" error
- Install LaTeX using Option 1 above
- Make sure `/Library/TeX/texbin` is in your PATH

### Figures not showing
- Make sure `report_figures/` directory exists with all PNG files
- Check that image paths in the .qmd file are correct

### Table formatting issues
- The YAML header includes LaTeX packages for better table formatting
- If tables look odd, you may need to install additional LaTeX packages

## What Gets Created

When you render successfully, you'll get:
- `OIL_FORECASTING_REPORT.pdf` - Your final PDF report
- `OIL_FORECASTING_REPORT.tex` - LaTeX source (if using LaTeX engine)

## Customization

To customize the PDF output, edit the YAML header in `OIL_FORECASTING_REPORT.qmd`:
- Change `fontsize` (currently 11pt)
- Adjust `geometry` margins
- Modify `linestretch` for line spacing
- Add/remove LaTeX packages in `include-in-header`

