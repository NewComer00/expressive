import oyaml
import numpy as np
from yamlcore import CoreLoader


def load_ustx(ustx_path):
    """Load a USTX (Vocal Synth format) file as a dictionary.

    Uses YAML parsing to extract the structure of a USTX file.

    Args:
        ustx_path (str): Path to the USTX file.

    Returns:
        dict: Parsed USTX data.
    """
    with open(ustx_path, "r", encoding="utf-8-sig") as u:
        ustx_str = u.read()
    # Use yamlcore.CoreLoader to support YAML1.2
    ustx_dict = oyaml.load(ustx_str, CoreLoader)
    return ustx_dict


def save_ustx(ustx_dict, ustx_path):
    """Save a USTX dictionary to a file, preserving order.

    Args:
        ustx_dict (dict): USTX data to save.
        ustx_path (str): Path to save the USTX file.
    """
    # Use oyaml to keep original order of USTX items
    output_str = oyaml.dump(ustx_dict, Dumper=oyaml.Dumper)
    with open(ustx_path, "w+", encoding="utf-8-sig") as o:
        o.write(output_str)


def edit_ustx_expression_curve(
    ustx_dict, ustx_track_number, expression, tick_seq, exp_seq
):
    if expression in ["dyn", "pitd", "tenc"]:
        track_idx = ustx_track_number - 1  # track index starts from 0
        track = ustx_dict["voice_parts"][track_idx]
        if "curves" not in track.keys():
            track["curves"] = []

        curves = track["curves"]
        exp = None
        for c in curves:
            if c["abbr"] == expression:
                exp = c
                break
        if exp is None:
            curves.append({"xs": [], "ys": [], "abbr": expression})
            exp = curves[-1]

        mask = ~np.isnan(exp_seq)
        exp["xs"] = tick_seq[mask].tolist()
        exp["ys"] = np.round(exp_seq[mask]).astype(int).tolist()

    else:
        raise ValueError(f"Unsupported expression type: {expression}")
