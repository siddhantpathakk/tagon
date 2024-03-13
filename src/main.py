from trainer import Trainer
from utils.main_utils import get_rank_results_paths, get_model_save_path
from utils.seed import seed_everything
from utils.data_utils import Data
from utils.loggers import set_up_logger
from utils.parser import parse_opt

if __name__ == "__main__":
    args = parse_opt()

    NUM_NEG = 1

    SAVE_MODEL_DIR, SAVE_MODEL_PATH, MODEL_SAVE_PATH, get_checkpoint_path = get_model_save_path(args)    
    RANK_RESULTS_FILE, RANK_RESULTS_DIR = get_rank_results_paths(args)

    logger = set_up_logger()
    logger.info(args)
    
    data = Data(args.data, args)

    seed_everything(seed=args.seed)

    trainer = Trainer(args, data, SAVE_MODEL_PATH)
    
    logger.info(f'Commencing training for {args.n_epoch} epochs')
    trainer.train()
    logger.info(f'Training finished')
    
    logger.info("Plotting results")
    trainer.plot_history()
    logger.info("Plotting finished")

    trainer.export_history()
    
    logger.info('Completed main file execution. Exiting...')
