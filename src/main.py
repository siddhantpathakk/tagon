from trainer import Trainer
from utils.data_utils import Data
from utils.main_utils import parse_opt, get_model_save_path, get_rank_results_paths, set_up_logger, seed_everything


if __name__ == "__main__":
    args = parse_opt()

    NUM_NEG = 1

    SAVE_MODEL_DIR, SAVE_MODEL_PATH, MODEL_SAVE_PATH, get_checkpoint_path = get_model_save_path(args)    
    RANK_RESULTS_FILE, RANK_RESULTS_DIR = get_rank_results_paths(args)

    logger = set_up_logger()
    logger.info(args)
    
    if args.negsampleeval == -1:
        # warn user that they are using the default value and that it may not be optimal
        logger.warning('Using default value for negsampleeval. This may not be optimal.')
    
    data = Data(args.data, args)

    seed_everything(seed=args.seed)

    trainer = Trainer(args, data, SAVE_MODEL_PATH)
    
    logger.info(f'Commencing training for {args.n_epoch} epochs')
    trainer.train()
    logger.info(f'Training finished. Completed main file execution. Exiting...')
