from .fmd_dataset import FaceManipulationDataset

class AttGANDataset(FaceManipulationDataset):
    def __init__(self, fake_dir, real_dir, mask_dir, high_quality_images_path=None, split='train'):
        super().__init__(fake_dir, real_dir, mask_dir, high_quality_images_path, split)

        # Optional: You can add or override anything specific to AttGAN
        print(f"[INFO] AttGANDataset initialized with {len(self.samples)} samples.")
