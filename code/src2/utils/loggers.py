import logging
import os

def set_logging_tool(output_dir):

    # check the dir, if it has any folder named 'runX" where X is a number, then return X
    def get_run_number(log_dir):
        runs = [int(run.split('run')[-1]) for run in os.listdir(log_dir) if run.startswith('run')]
        if len(runs) == 0:
            return 1
        return max(runs) + 1

    latest_run = get_run_number(output_dir)
    
    # create new directory for the run
    output_dir = os.path.join(output_dir, f'run{latest_run}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    log_file = os.path.join(output_dir, 'init.log')
    training_log_file = os.path.join(output_dir, 'training_log.log')

    # if the log file already exists, then delete it
    if os.path.exists(log_file):
        os.remove(log_file)
        
    if not os.path.isfile(log_file):
        open(log_file, "w+").close()
        
    if os.path.exists(training_log_file):
        os.remove(training_log_file)
    
    if not os.path.isfile(training_log_file):
        open(training_log_file, "w+").close()

    console_logging_format = f"%(levelname)s:\t%(module)s:\t%(message)s"
    file_logging_format = f"%(levelname)s:\t%(module)s:\t%(asctime)s:\t%(message)s"
    model_training_logging_format = f"%(asctime)s:\t%(message)s"
    
    # configure logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format=console_logging_format)

    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    model_training_logger = logging.getLogger('model_training')
    model_training_logger.setLevel(logging.INFO)
    model_training_handler = logging.FileHandler(training_log_file)
    model_training_logger.addHandler(model_training_handler)
    model_training_handler.setFormatter(logging.Formatter(model_training_logging_format))
    
    return latest_run, logger, model_training_logger