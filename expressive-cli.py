import argparse

from utils.gpu import add_cuda11_to_path
from expressive import process_expressions


def main():
    parser = argparse.ArgumentParser(
        description="Process USTX expressions from reference and UTAU WAV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General arguments
    parser.add_argument("-u", "--utau_wav", type=str, required=True, help="Path to the UTAU audio file")
    parser.add_argument("-r", "--ref_wav", type=str, required=True, help="Path to the reference audio file")
    parser.add_argument("-i", "--ustx_input", type=str, required=True, help="Path to the input USTX project file")
    parser.add_argument("-o", "--ustx_output", type=str, required=True, help="Path to save the processed USTX file")
    parser.add_argument("-t", "--track_number", type=int, required=True, help="Track number to apply expressions")
    
    # Expression selection
    parser.add_argument("-e", "--expression", type=str, action="append", required=True, choices=["dyn", "pitd"], 
                        help="Specify expressions to apply (e.g., --expression dyn --expression pitd)")

    # Group for Dyn-specific parameters
    dyn_group = parser.add_argument_group("Dynamic Expression (dyn)")
    dyn_group.add_argument("--dyn.align_radius", type=int, default=1, help="Alignment radius for 'dyn'")
    dyn_group.add_argument("--dyn.smoothness", type=int, default=2, help="Smoothness for 'dyn'")
    dyn_group.add_argument("--dyn.scaler", type=float, default=2.0, help="Scaler for 'dyn'")

    # Group for Pitd-specific parameters
    pitd_group = parser.add_argument_group("Pitch Deviation Expression (pitd)")
    pitd_group.add_argument("--pitd.confidence_utau", type=float, default=0.8, help="Confidence for UTAU in 'pitd'")
    pitd_group.add_argument("--pitd.confidence_ref", type=float, default=0.6, help="Confidence for reference in 'pitd'")
    pitd_group.add_argument("--pitd.align_radius", type=int, default=1, help="Alignment radius for 'pitd'")
    pitd_group.add_argument("--pitd.semitone_shift", type=float, default=None, help="Semitone shift for 'pitd'")
    pitd_group.add_argument("--pitd.smoothness", type=int, default=2, help="Smoothness for 'pitd'")
    pitd_group.add_argument("--pitd.scaler", type=float, default=2.0, help="Scaler for 'pitd'")

    args = parser.parse_args()

    expressions = []
    if "dyn" in args.expression:
        expressions.append({
            "expression": "dyn",
            "align_radius": getattr(args, "dyn.align_radius"),
            "smoothness": getattr(args, "dyn.smoothness"),
            "scaler": getattr(args, "dyn.scaler"),
        })
    if "pitd" in args.expression:
        expressions.append({
            "expression": "pitd",
            "confidence_utau": getattr(args, "pitd.confidence_utau"),
            "confidence_ref": getattr(args, "pitd.confidence_ref"),
            "align_radius": getattr(args, "pitd.align_radius"),
            "semitone_shift": getattr(args, "pitd.semitone_shift"),
            "smoothness": getattr(args, "pitd.smoothness"),
            "scaler": getattr(args, "pitd.scaler"),
        })

    process_expressions(
        args.utau_wav, args.ref_wav, args.ustx_input,
        args.ustx_output, args.track_number, expressions
    )

if __name__ == "__main__":
    add_cuda11_to_path()
    main()
