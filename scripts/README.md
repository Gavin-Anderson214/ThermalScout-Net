# Supporting tools

[← Project home](../README.md)

These tools are grouped here to keep the repository's top level focused. Run Python tools from the repository root using `py -3.12 -m scripts.NAME`.

| Task | Tool |
|---|---|
| Inspect validation errors | [analyze_validation.py](analyze_validation.py) |
| Check image-padding effects | [diagnose_padding.py](diagnose_padding.py) |
| Compare class-weight previews | [preview_class_weight.py](preview_class_weight.py) |
| Run the controlled class-weight experiment | [run_class_weight_experiment.py](run_class_weight_experiment.py) |
| Summarize completed class-weight experiments | [summarize_class_weight.py](summarize_class_weight.py) |
| Double-click Windows shortcuts | [launchers](launchers/) |

The launchers switch to the repository root before running, so they work from their new folder. The experiment runner starts training; the summarizer updates reports. Run those only when you intend those actions.

## Dataset preparation

birdsai_prepare.py inspects BIRDSAI frames and annotations and exports development training/validation data. Review existing previews before preparing again. Preserve original images and annotations.

[Back to Start Here](../START_HERE.md)

package_project.py builds and verifies the shareable ZIP under dist/ without datasets or model weights. Run it from any folder using its path.
