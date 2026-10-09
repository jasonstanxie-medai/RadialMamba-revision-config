from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans


class nnUNetTrainerUMambaEncCoord(nnUNetTrainer):
    """UMambaEnc 2D with AddCoordinates (x, y, optional r) before encoder stem."""

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise NotImplementedError(
                "nnUNetTrainerUMambaEncCoord only supports 2D. "
                "Use nnUNetTrainerUMambaEnc for 3D without coordinates."
            )
        model = get_umamba_enc_2d_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
            use_add_coordinates=True,
            add_coordinates_with_r=True,
        )
        print("UMambaEnc + AddCoordinates (x,y,r): {}".format(model))
        return model
