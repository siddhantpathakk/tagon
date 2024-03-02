from .utils.loggers import set_logging_tool
from .dataloader import DataCollector
from .trainer_v2 import Trainer
from .utils.seed import seed_everything
from .utils.parse import parse_opt

if __name__ == '__main__':
    config = parse_opt()
    seed_everything(config.seed)
    logger = set_logging_tool(log_file=config.log_file)
    
    logger.info(f'Using seed: {config.seed}')
    logger.info(f'Using device: {config.device}')
    
    logger.info(f'Commencing data collection for {config.dataset}')
    datacollector = DataCollector(config)
    train_part, test_part, info, edges = datacollector.prepare(verbose=config.verbose, logger=logger)
    
    logger.info(f'Commencing training for {config.epoch_num} epochs')
    trainer = Trainer(config=config,info=info,edges=edges)
    trainer.train(train_part=train_part,test_part=test_part)

    logger.info(f'Completed training for {config.epoch_num} epochs')