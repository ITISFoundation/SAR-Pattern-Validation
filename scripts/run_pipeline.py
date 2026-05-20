from sar_pattern_validation.workflow_config import WorkflowConfig
from sar_pattern_validation.workflows import _complete_workflow


def main():
    # input parameters
    measured_dir = "data/example/"
    measured_file = "measured_sSAR1g.csv"
    reference_file = "reference_sSAR1g.csv"
    power_level_dbm = 30.0

    # configuration
    output_dir = "results/"
    config = WorkflowConfig(
        measured_file_path=measured_dir + measured_file,
        reference_file_path=measured_dir + reference_file,
        reference_image_save_path=output_dir + "reference_image.png",
        measured_image_save_path=output_dir + "measured_image.png",
        aligned_meas_save_path=output_dir + "aligned_meas_image.png",
        registered_image_save_path=output_dir + "registration.png",
        gamma_comparison_image_path=output_dir + "gamma.png",
        power_level_dbm=power_level_dbm,
    )

    result = _complete_workflow(config)

    # output parameters
    passed = result.passed_pixel_count == result.evaluated_pixel_count
    pass_rate = result.pass_rate_percent
    measured_pssar = result.measured_pssar
    reference_pssar = result.reference_pssar
    scaling_error = result.scaling_error

    # print outputs to console
    if passed:
        print("Test passed.")
    else:
        print("Test failed.")
    print(f"Pass rate = {pass_rate:.1f}%")
    print(
        f"Measured psSAR at {power_level_dbm:.1f} dBm: {measured_pssar * (10 ** ((power_level_dbm - 30) / 10.0)):.3f} W/kg"
    )
    print(f"Measured psSAR at 30 dBm: {measured_pssar:.2f} W/kg")
    print(f"Reference psSAR: {reference_pssar:.2f} W/kg")
    print(f"Scaling error: {100 * scaling_error:.2f} %")
    return


if __name__ == "__main__":
    main()
