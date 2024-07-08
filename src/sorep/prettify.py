"""Functions for prettifying strings for LaTeX and Unicode output."""

import re

__all__ = ("latex_kpoint_label", "unicode_kpoint_label", "latex_chemical_formula", "unicode_chemical_formula")


def latex_kpoint_label(label: str) -> str:
    """Prettify a k-point label by replacing keywords with LaTeX strings.

    Args:
        label (str): k-point label (e.g. "GAMMA")

    Returns:
        str: LaTeXified string (e.g. r"$\\Gamma$")
    """
    # Replace spelled-out greek letters with LaTeX symbols
    # yapf: disable
    label = (
        label
            .replace('GAMMA', r'$\Gamma$')
            .replace('DELTA', r'$\Delta$')
            .replace('LAMBDA', r'$\Lambda$')
            .replace('SIGMA', r'$\Sigma$')
    )
    # yapf: enable
    # Replace underscore-numerals with LaTeX subscripts
    label = re.sub(r"_(.?)", r"$_{\1}$", label)
    return label


def unicode_kpoint_label(label: str) -> str:
    """Prettify a k-point label by replacing keywords with Unicode characters.

    Args:
        label (str): k-point label(e.g. "GAMMA")

    Returns:
        str: Unicode string (e.g. "Γ")
    """
    # Replace spelled-out greek letters with Unicode characters
    # yapf: disable
    label = (
        label
            .replace('GAMMA', "Γ")
            .replace('DELTA', "Δ")
            .replace('LAMBDA', "Λ")
            .replace('SIGMA', "Σ")
            .replace('_0', '\u2080')
            .replace('_1', '\u2081')
            .replace('_2', '\u2082')
            .replace('_3', '\u2083')
            .replace('_4', '\u2084')
            .replace('_5', '\u2085')
            .replace('_6', '\u2086')
            .replace('_7', '\u2087')
            .replace('_8', '\u2088')
            .replace('_9', '\u2089')

    )
    return label


def latex_chemical_formula(formula: str) -> str:
    """Prettify a chemical formula by replacing numbers with LaTeX subscripts.

    Args:
        formula (str): Chemical formula.

    Returns:
        str: Chemical formula with LaTeX subscripts.
    """
    formula = re.sub(r"(\d+)", r"$_{\1}$", formula)
    return formula


def unicode_chemical_formula(formula: str) -> str:
    """Prettify a chemical formula by replacing numbers with Unicode subscripts.

    Args:
        formula (str): Chemical formula.

    Returns:
        str: Chemical formula with Unicode subscripts.
    """
    formula = (
        formula.replace("0", "\u2080")
        .replace("1", "\u2081")
        .replace("2", "\u2082")
        .replace("3", "\u2083")
        .replace("4", "\u2084")
        .replace("5", "\u2085")
        .replace("6", "\u2086")
        .replace("7", "\u2087")
        .replace("8", "\u2088")
        .replace("9", "\u2089")
    )
    return formula
