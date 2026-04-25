"""
Configuration options for the NeoNude pipeline.
"""


class Options:
    """Configuration for GAN model and data loading."""

    def __init__(self):
        # Normalization
        self.norm = "batch"

        # Dropout
        self.use_dropout = False

        # Data type: 8, 16, or 32 bit
        self.data_type = 32

        # Input/output
        self.batch_size = 1
        self.input_nc = 3
        self.output_nc = 3

        # Data loading
        self.serial_batches = True
        self.n_threads = 1
        self.max_dataset_size = 1

        # Generator architecture
        self.net_g = "global"
        self.ngf = 64
        self.n_downsample_global = 4
        self.n_blocks_global = 9
        self.n_blocks_local = 0
        self.n_local_enhancers = 0
        self.niter_fix_global = 0

        # Paths (set per phase)
        self.checkpoints_dir = ""
        self.dataroot = ""

    def update_for_phase(self, phase: str):
        """Update options based on the current pipeline phase.

        Args:
            phase: One of 'correct_to_mask', 'maskref_to_maskdet',
                   or 'maskfin_to_nude'.
        """
        phase_checkpoints = {
            "correct_to_mask": "checkpoints/cm.lib",
            "maskref_to_maskdet": "checkpoints/mm.lib",
            "maskfin_to_nude": "checkpoints/mn.lib",
        }

        if phase in phase_checkpoints:
            self.checkpoints_dir = phase_checkpoints[phase]
