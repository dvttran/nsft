import sys
# NOTE: use sys.path.append if running this script outside the root directory
from nsft.runners import get_runner
import torch

def main(
        config_file,
        output_path,
        surfaces_name,
        rgbs_name,
        depths_name,
        materials,
        device,
        **kwargs
):
    runner = get_runner(config_file=config_file)
    runner.run(
        output_path=output_path,
        surfaces_name=surfaces_name,
        rgbs_name=rgbs_name,
        depths_name=depths_name,
        materials=materials,
        device=device,
        **kwargs
    )


if __name__ == "__main__":
    # config_file = "configs/ICCV2025/Phi_SfT/default.yaml"
    config_file = "configs/ICCV2025/KINECT_PAPER/default.yaml"
    if len(sys.argv) > 1:
        for args in sys.argv[1:]:
            if "config_file" in args:
                config_file = args.split("=")[1]
    config_name = config_file.split("/")[-1].split(".")[0]
    print(f"Config name: {config_name}")

    output_path = f"outputs/{config_name}"
    if len(sys.argv) > 1:
        for args in sys.argv[1:]:
            if "output_path" in args:
                output_path = args.split("=")[1]
                break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(
        config_file=config_file,
        output_path=output_path,
        surfaces_name="00",
        rgbs_name="rgb",
        depths_name=None,
        materials=False,
        device=device,
    )
