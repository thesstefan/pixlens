from pixlens.utils import utils
from pathlib import Path
class Config:
    def __init__(self):
        # experiment specifics
        self.dist_url = 'tcp://127.0.0.1:10002'
        self.num_gpu = 8
        self.name = 'open-edit'
        self.checkpoints_dir = Path(utils.get_cache_dir() / 'models--openedit/checkpoints')
        self.vocab_path = Path(utils.get_cache_dir() / 'models--openedit/vocab/conceptual_vocab.pkl')
        self.vse_enc_path = Path(utils.get_cache_dir() / 'models--openedit/checkpoints/conceptual_model_best.pth.tar') 
        self.edge_model_path = Path(utils.get_cache_dir() / 'models--openedit/checkpoints/bdcn_pretrained_on_bsds500.pth')
        self.model = 'OpenEdit'
        self.norm_G = 'spectralsync_batch'
        self.norm_D = 'spectralinstance'
        self.phase = 'train'
        
        # input/output sizes
        self.batchSize = 8
        self.img_size = 224
        self.output_nc = 3
        
        # for setting inputs
        self.dataroot = '/pixlens/editing/impl/open_edit/datasets/conceptual/'
        self.dataset_mode = 'conceptual'
        self.serial_batches = False
        self.nThreads = 4
        self.load_from_opt_file = False
        
        # for displays
        self.display_winsize = 256
        
        # for generator
        self.netG = 'openedit'
        self.ngf = 64
        self.init_type = 'xavier'
        self.init_variance = 0.02
        
        # for encoder
        self.netE = 'resnetbdcn'
        self.edge_nc = 1
        self.edge_level = 41
        self.edge_tanh = False
        
        # for image-specific finetuning
        self.reg_weight = 1e-4
        self.perturbation = False
        self.manipulation = False
        self.img_path = None
        self.ori_cap = None
        self.new_cap = None
        self.global_edit = False
        self.alpha = 5
        self.optimize_iter = 50

        # Training options
        self.print_freq = 100
        self.save_epoch_freq = 10
        self.no_html = False
        self.tf_log = False
        self.continue_train = False
        self.which_epoch = 'latest'
        self.niter = 50
        self.niter_decay = 50
        self.optimizer = 'adam'
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lr = 0.0002
        self.D_steps_per_G = 1
        self.G_steps_per_D = 1
        self.ndf = 64
        self.lambda_feat = 10.0
        self.lambda_vgg = 10.0
        self.no_ganFeat_loss = False
        self.no_vgg_loss = False
        self.no_l1feat_loss = False
        self.l1pix_loss = False
        self.gan_mode = 'hinge'
        self.netD = 'multiscale'
        self.no_TTUR = False
        self.no_disc = False

        self.isTrain = True
