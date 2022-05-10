from typing import Any, Dict

from monai.networks.nets import UNet as MONAI_UNet


class UNet(MONAI_UNet):
    """ Inheiriting MONAI UNET passing config agruments

    Documetnation located: https://docs.monai.io/en/stable/_modules/monai/networks/nets/unet.html
    """
    def __init__(
        self,
        **kwargs: Dict[str, Any]
    ):
        super().__init__(**kwargs)
