from utils.loggers import set_logging_tool
from dataloader import DataCollector
from trainer import Trainer
from utils.seed import seed_everything
from utils.parse import parse_opt

if __name__ == '__main__':
    runCount, logger, ml_logger = set_logging_tool("/home/FYP/siddhant005/fyp/runs")
    
    config = parse_opt(runCount)
    seed_everything(config.seed)
    
    logger.info(f'Outputs will be saved in: runs/run{runCount}')
    logger.info(f'Using seed: {config.seed}')
    logger.info(f'Using device: {config.device}')
    
    logger.info(f'Commencing data collection for {config.dataset}')
    datacollector = DataCollector(config)
    train_part, test_part, info, edges = datacollector.prepare(verbose=config.verbose, logger=logger)

    trainer = Trainer(config=config,info=info,edges=edges, logger=ml_logger, device=config.device)
    logger.info(f'Commencing training for {config.epoch_num} epochs.')
    trainer.train(train_part=train_part,test_part=test_part)
    logger.info(f'Completed training for {config.epoch_num} epochs')
    
    trainer.save_model(path_name=f'/home/FYP/siddhant005/fyp/runs/run{runCount}/model.pth')
    logger.info(f'Model saved at runs/run{runCount}/model.pth')
    logger.info(f'Completed run {runCount}')
