"""Descriptor-to-text templates; target uptake is never included in prompts."""
import numpy as np
from .data import STRUCTURES

LABELS = ["BET surface area", "Total pore volume", "Micropore volume", "Ultramicropore volume", "Mesopore volume"]
MISSING_SENTENCES = ["BET surface area is missing.", "Total pore volume is unavailable.",
                     "Micropore volume is unavailable.", "Ultramicropore volume is unknown.",
                     "Mesopore volume is unavailable."]

def make_prompt(values, flags):
    values = np.asarray(values, float)
    if len(values) != 11 or len(flags) != 5:
        raise ValueError("Expected 11 descriptors and 5 structural mask flags")
    sentences = [f"{name} content is {value:.2f} wt%." for name, value in zip(
        ["Carbon", "Hydrogen", "Nitrogen", "Oxygen"], values[:4])]
    for j, (label, col) in enumerate(zip(LABELS, STRUCTURES)):
        if flags[j]:
            sentences.append(MISSING_SENTENCES[j])
        else:
            unit = "m2/g" if j == 0 else "cm3/g"
            sentences.append(f"{label} is {values[4+j]} {unit}.")
    sentences.append(f"The adsorption experiment is conducted at a temperature of {values[9]} K and a pressure of {values[10]} bar.")
    return " ".join(sentences)
