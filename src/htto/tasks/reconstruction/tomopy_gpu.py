from cupy import ndarray
from tomopy import astra, recon


def reconstruct(
    data: ndarray, angles_radians: ndarray, rot_center: float, use_GPU: int
):
    """Perform a reconstruction using tomopy's astra recon function.

    Args:
        data: The sinograms to reconstruct.
        angles_radians: A numpy array of angles in radians.
        rot_center: The rotational center.
        use_GPU: The ID of the GPU to use.

    Returns:
        ndarray: A numpy array containing the reconstructed volume.
    """
    return recon(
        data,
        angles_radians,
        center=rot_center,
        algorithm=astra,
        options={
            "method": "FBP_CUDA",
            "proj_type": "cuda",
            "gpu_list": [use_GPU],
        },
        ncore=1,
    )
